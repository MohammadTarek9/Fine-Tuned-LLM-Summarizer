# Baseline vs Fine-Tuned ROUGE Comparison

## Baseline (Before Fine-Tuning)

Generated at: 2026-04-17 12:29:29 UTC

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Samples | Notes |
|---|---:|---:|---:|---:|---|
| FLAN-T5 Base (Zero-shot) | 0.3265 | 0.0981 | 0.2237 | 20 | temp=0.7, top_k=50, top_p=0.95, errors=0 |
| FLAN-T5 Base (Few-shot) | 0.0739 | 0.0000 | 0.0597 | 20 | 2 demos, temp=0.7, errors=0 |

## Notes
- Dataset: cnn_dailymail test subset (20 samples)
- Base model: google/flan-t5-base
- Fine-tuned section will be appended by notebook 03