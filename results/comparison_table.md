# Baseline vs Fine-Tuned ROUGE Comparison

## Baseline (Before Fine-Tuning)

Generated at: 2026-04-17 20:59:32 UTC

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Samples | Notes |
|---|---:|---:|---:|---:|---|
| FLAN-T5 Base (Zero-shot) | 0.3610 | 0.1464 | 0.2466 | 70 | temp=0.0, top_k=None, top_p=None, errors=0 |
| FLAN-T5 Base (Few-shot) | 0.3621 | 0.1473 | 0.2476 | 70 | 2 demos, temp=0.0, errors=0 |

## Notes
- Dataset: cnn_dailymail test subset (70 samples)
- Base model: google/flan-t5-base
- Fine-tuned section will be appended by notebook 03