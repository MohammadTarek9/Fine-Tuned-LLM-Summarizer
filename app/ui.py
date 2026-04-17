"""Gradio app for side-by-side base vs fine-tuned summarization."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import gradio as gr
from src.model import DomainSummarizer, GenerationParams

BASE_MODEL = DomainSummarizer(model_name="google/flan-t5-base")
FINETUNED_MODEL: Optional[DomainSummarizer] = None


def load_models() -> None:
    """load base model and best-effort fine-tuned model for UI interactions"""
    global FINETUNED_MODEL

    BASE_MODEL.load()

    adapter_dir = Path("./lora-adapter")
    if adapter_dir.exists():
        try:
            FINETUNED_MODEL = DomainSummarizer(
                model_name="google/flan-t5-base",
                adapter_path=str(adapter_dir),
            )
            FINETUNED_MODEL.load()
        except Exception:
            FINETUNED_MODEL = None


def summarize_side_by_side(
    text: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
) -> Tuple[str, str, str]:
    """generate base and fine-tuned summaries with shared generation settings"""
    if not text or len(text.strip()) < 10:
        return "", "", "Please provide at least 10 characters of text."

    params = GenerationParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    try:
        base_summary = BASE_MODEL.summarize(text, generation=params)
    except Exception as exc:
        base_summary = f"Base model error: {exc}"

    if FINETUNED_MODEL is not None:
        try:
            ft_summary = FINETUNED_MODEL.summarize(text, generation=params)
        except Exception as exc:
            ft_summary = f"Fine-tuned model error: {exc}"
    else:
        ft_summary = "Fine-tuned model not loaded. Train and save adapter to ./lora-adapter"

    return base_summary, ft_summary, "Done"


def build_ui() -> gr.Blocks:
    """create and return the Gradio Blocks interface"""
    with gr.Blocks(title="Domain Summarizer Demo") as demo:
        gr.Markdown("# Domain-Adaptive Summarizer")
        gr.Markdown("Compare base FLAN-T5 vs LoRA fine-tuned adapter on the same input.")

        with gr.Row():
            input_text = gr.Textbox(
                label="Article Text",
                lines=12,
                placeholder="Paste article text here...",
            )

        with gr.Row():
            temperature = gr.Slider(0.01, 2.0, value=0.7, step=0.01, label="Temperature")
            top_k = gr.Slider(0, 100, value=50, step=1, label="Top-k")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="Top-p")
            max_new_tokens = gr.Slider(16, 512, value=128, step=8, label="Max new tokens")

        run_btn = gr.Button("Summarize")

        with gr.Row():
            base_output = gr.Textbox(label="Base Model Summary", lines=8)
            ft_output = gr.Textbox(label="Fine-Tuned Model Summary", lines=8)

        status = gr.Textbox(label="Status", lines=1)

        run_btn.click(
            fn=summarize_side_by_side,
            inputs=[input_text, temperature, top_k, top_p, max_new_tokens],
            outputs=[base_output, ft_output, status],
        )

    return demo


if __name__ == "__main__":
    load_models()
    ui = build_ui()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False)