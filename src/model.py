"""Reusable model utilities for domain-adaptive summarization."""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



@dataclass
class FewShotExample:
    article: str
    summary: str


@dataclass
class GenerationParams:
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_new_tokens: int = 256

    def use_sampling(self) -> bool:
        """Decide whether to sample or use greedy decoding"""
        return self.temperature is not None and self.temperature > 0

    def sanitized(self) -> GenerationParams:
        return GenerationParams(
            temperature=max(0.0, float(self.temperature)),
            top_k=None if self.top_k in [None, 0] else int(self.top_k),
            top_p=None if self.top_p in [None, 0] else float(min(max(self.top_p, 0.1), 1.0)),
            max_new_tokens=int(min(self.max_new_tokens, 512)),
        )


@dataclass
class DomainSummarizer:
    model_name: str = "google/flan-t5-base"
    adapter_path: Optional[str] = None
    device: Optional[str] = None
    seed: int = 42

    max_input_tokens: int = 512

    max_article_chars: int = 2000
    max_demo_article_chars: int = 200
    max_demo_summary_chars: int = 80

    tokenizer: Optional[AutoTokenizer] = field(default=None, init=False)
    model: Optional[AutoModelForSeq2SeqLM] = field(default=None, init=False)

    def load(self) -> None:
        set_seed(self.seed)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            dtype=dtype,
        )

        # Override max_input_tokens with the model's own declared limit if it
        # is smaller than what was configured — prevents positional embedding
        # errors on models like flan-t5-base whose true limit is 512.
        model_max = getattr(self.model.config, "n_positions", None) or \
                    getattr(self.model.config, "max_position_embeddings", None) or \
                    getattr(self.tokenizer, "model_max_length", None)
        if model_max and model_max < self.max_input_tokens:
            self.max_input_tokens = model_max

        if self.adapter_path:
            adapter_dir = Path(self.adapter_path)
            if not adapter_dir.exists():
                raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")
            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))

        self.model.to(self.device)
        self.model.eval()

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")


    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0].strip() + " ..."

    def _token_len(self, text: str) -> int:
        """Count tokens without triggering max-length warnings."""
        if self.tokenizer is None:
            # Rough fallback when tokenizer is unavailable.
            return len(text) // 3
        return len(self.tokenizer.tokenize(text))

    def build_zero_shot_prompt(self, article: str) -> str:
        article = self._truncate_text(article, self.max_article_chars)

        return (
            "Summarize the following news article in 2-3 sentences.\n\n"
            f"Article:\n{article}\n\n"
            "Summary:"
        )

    def build_few_shot_prompt(
        self,
        article: str,
        examples: List[FewShotExample],
    ) -> str:
        article = self._truncate_text(article, self.max_article_chars)

        # Measure the fixed overhead: instruction + query article + "Summary:" label.
        # This is everything in the prompt that isn't the demonstration blocks.
        # The demo budget is whatever token space remains after this fixed overhead.
        fixed_frame = (
            "Summarize the following news article in 2-3 sentences.\n\n"
            f"Article:\n{article}\n\n"
            "Summary:"
        )

        fixed_tokens = self._token_len(fixed_frame)

        demo_budget = max(0, self.max_input_tokens - fixed_tokens - 8)  # 8-token safety margin

        demo_blocks = []
        tokens_used = 0
        for ex in examples:
            demo_article = self._truncate_text(ex.article, self.max_demo_article_chars)
            demo_summary = self._truncate_text(ex.summary, self.max_demo_summary_chars)
            block = f"Article:\n{demo_article}\n\nSummary:\n{demo_summary}\n"

            block_tokens = self._token_len(block)

            if tokens_used + block_tokens > demo_budget:
                break  # Skip this demo rather than overflow the context

            demo_blocks.append(block)
            tokens_used += block_tokens

        demonstrations = "\n".join(demo_blocks)

        return (
            "Summarize the following news article in 2-3 sentences.\n\n"
            f"{demonstrations}\n"
            f"Article:\n{article}\n\n"
            "Summary:"
        )

    def summarize(
        self,
        article: str,
        generation: Optional[GenerationParams] = None,
        few_shot_examples: Optional[List[FewShotExample]] = None,
    ) -> str:
        self._ensure_loaded()

        if not article.strip():
            raise ValueError("Input text is empty.")

        generation = (generation or GenerationParams()).sanitized()

        use_few_shot = bool(few_shot_examples)

        prompt = (
            self.build_few_shot_prompt(article, few_shot_examples)
            if use_few_shot
            else self.build_zero_shot_prompt(article)
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        generate_kwargs = {
            "max_new_tokens": generation.max_new_tokens,
            "num_beams": 1,
        }

        if generation.use_sampling():
            generate_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": generation.temperature,
                }
            )

            if generation.top_k is not None:
                generate_kwargs["top_k"] = generation.top_k

            if generation.top_p is not None:
                generate_kwargs["top_p"] = generation.top_p
        else:
            generate_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                **generate_kwargs,
            )

        summary = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()

        return summary

    def summarize_batch(
        self,
        articles: List[str],
        generation: Optional[GenerationParams] = None,
        few_shot_examples: Optional[List[FewShotExample]] = None,
    ) -> List[str]:
        return [
            self.summarize(
                article=text,
                generation=generation,
                few_shot_examples=few_shot_examples,
            )
            for text in articles
        ]

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()