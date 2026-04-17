"""Kickstart LoRA fine-tuning script for FLAN-T5 summarization."""
from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

def set_global_seed(seed: int = 42) -> None:
    """set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class TrainConfig:
    """configuration for LoRA fine-tuning"""
    model_name: str = "google/flan-t5-base"
    dataset_name: str = "cnn_dailymail"
    dataset_config: str = "3.0.0"
    max_train_samples: int = 5000
    max_source_length: int = 512
    max_target_length: int = 128
    output_dir: str = "./lora-adapter"
    run_name: str = "flan-t5-lora-cnn-dm"
    learning_rate: float = 2e-4
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    logging_steps: int = 20
    save_steps: int = 250
    seed: int = 42
    use_wandb: bool = True
    wandb_project: str = "domain-summarizer"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = None

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = ["q", "v"]
    

def build_preprocess_fn(tokenizer, cfg: TrainConfig):
    """create preprocessing function for tokenizer.map"""
    def preprocess(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        prompts = [f"Summarize the following article:\n\n{x}" for x in batch["article"]]

        model_inputs = tokenizer(
            prompts,
            max_length=cfg.max_source_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=batch["highlights"],
            max_length=cfg.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def train(cfg: TrainConfig) -> None:
    """run full LoRA fine-tuning and save adapter weights"""
    set_global_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name,
            config=cfg.__dict__,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch_dtype,
    )

    # gradient checkpointing trades extra compute for lower memory usage on T4
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split="train")
    dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_train_samples))

    preprocess_fn = build_preprocess_fn(tokenizer, cfg)
    tokenized = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=1,
        fp16=(device == "cuda"),
        bf16=False,
        report_to=["wandb"] if cfg.use_wandb else [],
        seed=cfg.seed,
        dataloader_pin_memory=(device == "cuda"),
        remove_unused_columns=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    try:
        trainer.train()
        model.save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"Saved LoRA adapter and tokenizer to: {cfg.output_dir}")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM during training. Reduce batch size, source length, "
                "or max_train_samples."
            ) from exc
        raise
    finally:
        if cfg.use_wandb:
            wandb.finish()

def parse_args() -> TrainConfig:
    """parse CLI arguments for common overrides"""
    parser = argparse.ArgumentParser(description="LoRA fine-tune FLAN-T5 on cnn_dailymail.")
    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="./lora-adapter")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    use_wandb = True
    if args.use_wandb:
        use_wandb = True
    if args.no_wandb:
        use_wandb = False

    return TrainConfig(
        max_train_samples=args.max_train_samples,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        use_wandb=use_wandb,
    )

if __name__ == "__main__":
    config = parse_args()
    train(config)