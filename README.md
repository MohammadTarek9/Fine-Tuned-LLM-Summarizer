# Domain-Adaptive Text Summarizer with Fine-Tuning Pipeline

## Project Overview
This project demonstrates an end-to-end workflow for adapting a large language model to a summarization task and comparing performance before and after fine-tuning.

The core model is `google/flan-t5-base`. The workflow includes:
- baseline inference with prompt engineering
- parameter-efficient fine-tuning (LoRA via PEFT)
- ROUGE-based evaluation and comparison
- generation configuration experiments
- deployment with FastAPI and Gradio

The design is optimized for Google Colab free tier (T4 GPU) with memory-aware defaults and reproducibility controls.

## Architecture
The project is organized into five layers:

1. Data Layer
- Uses `cnn_dailymail` from Hugging Face Datasets.
- Selects compact subsets during iteration for speed and lower VRAM usage.

2. Modeling Layer
- Base model loading and generation utilities are centralized in `src/model.py`.
- Supports zero-shot and few-shot prompting styles.
- Supports configurable decoding parameters: `temperature`, `top_k`, `top_p`, `max_new_tokens`.

3. Training Layer
- LoRA fine-tuning pipeline in `src/train.py`.
- Uses PEFT adapters instead of full model updates to reduce compute and memory requirements.
- Logs experiments to Weights and Biases.

4. Evaluation Layer
- ROUGE metrics in `src/evaluate.py`.
- Side-by-side comparison workflow for baseline vs fine-tuned model.

5. Serving Layer
- FastAPI endpoint for programmatic access.
- Gradio app for interactive testing.
