"""Evaluation utilities for summarization quality metrics."""

from __future__ import annotations
from typing import Dict, List
from rouge_score import rouge_scorer


def validate_pairs(references: List[str], predictions: List[str]) -> None:
    """validate that references and predictions are aligned before scoring"""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length.")
    if len(references) == 0:
        raise ValueError("references and predictions must not be empty.")


def compute_rouge_single(reference: str, prediction: str) -> Dict[str, float]:
    """compute ROUGE-1, ROUGE-2, ROUGE-L for one pair"""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference.strip(), prediction.strip())
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def compute_rouge_batch(references: List[str], predictions: List[str]) -> Dict[str, float]:
    """compute mean ROUGE metrics over many reference-prediction pairs"""
    validate_pairs(references, predictions)

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref, pred in zip(references, predictions):
        row = compute_rouge_single(ref, pred)
        totals["rouge1"] += row["rouge1"]
        totals["rouge2"] += row["rouge2"]
        totals["rougeL"] += row["rougeL"]

    count = float(len(references))
    return {
        "rouge1": round(totals["rouge1"] / count, 4),
        "rouge2": round(totals["rouge2"] / count, 4),
        "rougeL": round(totals["rougeL"] / count, 4),
    }


def format_comparison_row(
    model_label: str,
    metrics: Dict[str, float],
    sample_count: int,
    notes: str = "",
) -> Dict[str, str]:
    """create a normalized table row dictionary for markdown export"""
    return {
        "model": model_label,
        "rouge1": f"{metrics['rouge1']:.4f}",
        "rouge2": f"{metrics['rouge2']:.4f}",
        "rougeL": f"{metrics['rougeL']:.4f}",
        "samples": str(sample_count),
        "notes": notes,
    }


def to_markdown_table(rows: List[Dict[str, str]]) -> str:
    """render comparison rows as a markdown table string"""
    if not rows:
        raise ValueError("rows must not be empty.")

    header = "| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Samples | Notes |"
    sep = "|---|---:|---:|---:|---:|---|"

    body = [
        f"| {r['model']} | {r['rouge1']} | {r['rouge2']} | {r['rougeL']} | {r['samples']} | {r['notes']} |"
        for r in rows
    ]
    return "\n".join([header, sep] + body)