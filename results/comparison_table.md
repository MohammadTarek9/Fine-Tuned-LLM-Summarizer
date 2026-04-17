# Baseline vs Fine-Tuned ROUGE Comparison

## Baseline (Before Fine-Tuning)

Generated at: 2026-04-17 20:37:17 UTC

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Samples | Notes |
|---|---:|---:|---:|---:|---|
| FLAN-T5 Base (Zero-shot) | 0.3300 | 0.1316 | 0.2324 | 20 | temp=0.0, top_k=None, top_p=None, errors=0 |
| FLAN-T5 Base (Few-shot) | 0.3300 | 0.1316 | 0.2324 | 20 | 2 demos, temp=0.0, errors=0 |

## Notes
- Dataset: cnn_dailymail test subset (20 samples)
- Base model: google/flan-t5-base
- Fine-tuned section will be appended by notebook 03