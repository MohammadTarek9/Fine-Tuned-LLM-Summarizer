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
    """set a random seed for reproducability across CPU and GPU"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class FewShotExample:
    """demonstration pair for few-shot prompting"""
    article: str
    summary: str

@dataclass
class GenerationParams:
    """inference hyperparameters"""
    temperature: float = 1
    top_k: int = 50
    top_p: float = 0.95
    max_new_tokens: int = 200

    def clip_values(self) -> GenerationParams:
        """clip inference values to safe ranges"""
        return GenerationParams(
            temperature=float(min(max(self.temperature, 0.01), 2.0)),
            top_k=int(max(self.top_k, 0)),
            top_p=float(min(max(self.top_p, 0.1), 1.0)),
            max_new_tokens=int(min(max(self.max_new_tokens, 16), 512)),
        )
    
@dataclass
class DomainSummarizer:
    """inference wrapper for base Flan-T5
    and optional LoRA-adapted model
    """
    model_name: str = "google/flan-t5-base"
    adapter_path: Optional[str] = None
    device: Optional[str] = None
    seed: int = 42
    max_input_tokens = 1024
    tokenizer: Optional[AutoTokenizer] = field(default = None, init = False)
    model: Optional[AutoModelForSeq2SeqLM] = field(default = None, init = False)

    def load(self) -> None:
        """load tokenizer/model and optionally attach LoRA adapter"""
        set_seed(self.seed)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            )

            if self.adapter_path:
                adapter_dir = Path(self.adapter_path)
                if not adapter_dir.exists():
                    raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")
                self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))

            # move all model params to the device
            self.model.to(self.device)
            self.model.eval()

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load summarizer model '{self.model_name}'"
            ) from exc
        
    def ensure_loaded(self) -> None:
        """ensure model and tokenizer are loaded"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        
    def build_zero_shot_prompt(self, article: str) -> str:
        """build a prompt for zero-shot inference"""
        article = article.strip()
        prompt = f"""
        Summarize the following news article in 2-3 sentences.\n
        Article:\n{article}\n\nSummary:
        """
        return prompt

    def build_few_shot_prompt(self, article: str, examples: List[FewShotExample]) -> str:
        """build a prompt for few-shot inference"""
        article = article.strip()
        demo_blocks = []
        for ex in examples:
            demo_blocks.append(f"Article:\n{ex.article.strip()}\n\nSummary:\n{ex.summary.strip()}\n")
            demonstrations = "\n".join(demo_blocks)
        
        prompt = f"""
        You are a concise summarization assistant.\n
        Use the style of the examples.\n\n
        {demonstrations}\n
        Article:\n
        {article}\n\n
        Summary:\n
        """
        return prompt
    




