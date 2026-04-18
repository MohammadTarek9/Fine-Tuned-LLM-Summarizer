# Baseline vs Fine-Tuned ROUGE Comparison

## Baseline (Before Fine-Tuning)

Generated at: 2026-04-17 20:59:32 UTC

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Samples | Notes |
|---|---:|---:|---:|---:|---|
| FLAN-T5 Base (Zero-shot) | 0.3610 | 0.1464 | 0.2466 | 70 | temp=0.0, top_k=None, top_p=None, errors=0 |
| FLAN-T5 Base (Few-shot) | 0.3621 | 0.1473 | 0.2476 | 70 | 2 demos, temp=0.0, errors=0 |


## Fine-Tuned Evaluation (After LoRA)

Generated at: 2026-04-18 15:12:26 UTC

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Samples | Notes |
|---|---:|---:|---:|---:|---|
| FLAN-T5 Base (Post-Train Eval) | 0.3389 | 0.1203 | 0.2321 | 100 | temp=0.7, errors=0 |
| FLAN-T5 + LoRA (Fine-tuned) | 0.3645 | 0.1413 | 0.2479 | 100 | adapter=lora-adapter, temp=0.7, errors=0 |

## Improvement Summary (Fine-tuned - Base)
- ROUGE-1 delta: +0.0256
- ROUGE-2 delta: +0.0210
- ROUGE-L delta: +0.0158
