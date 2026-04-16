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

    def __post__init__(self) -> None:
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
