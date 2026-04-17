"""FastAPI starter for summarization inference."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.model import DomainSummarizer, GenerationParams

app = FastAPI(title="Domain Summarizer API", version="1.0.0")

BASE_MODEL = DomainSummarizer(model_name = "google/flan-t5-base")
FINETUNED_MODEL: Optional[DomainSummarizer] = None

class SummarizeRequest(BaseModel):
    """input schema for summarize endpoint"""
    text: str = Field(..., min_length=10, description="Article text to summarize.")
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.95, ge=0.1, le=1.0)
    max_new_tokens: int = Field(128, ge=16, le=512)
    use_finetuned: bool = Field(False, description="Use LoRA adapter if available.")

class SummarizeResponse(BaseModel):
    """output schema for summarize endpoint"""
    summary: str
    model_used: str

@app.on_event("startup")
def startup_event() -> None:
    """load base model once at startup and load adapter if available"""
    global FINETUNED_MODEL
    try:
        BASE_MODEL.load()
    except Exception as exc:
        raise RuntimeError("Failed to load base summarizer on startup.") from exc
    
    adapter_dir = Path("./lora-adapter")
    if adapter_dir.exists():
        try:
            FINETUNED_MODEL = DomainSummarizer(model_name="google/flan-t5-base", adapter_path=str(adapter_dir))
        except Exception:
            # serve base model if adapter path fails
            FINETUNED_MODEL = None


@app.get("/health")
def health() -> dict:
    """health endpoint for quick service checks"""
    return {
        "status": "ok",
        "base_model_loaded": BASE_MODEL.model is not None,
        "finetuned_model_loaded": FINETUNED_MODEL is not None and FINETUNED_MODEL.model is not None,
    }

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    """summarize text with base or fine-tuned model"""
    generation = GenerationParams(
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        max_new_tokens=payload.max_new_tokens
    )

    model = BASE_MODEL
    model_label = "base"
    if payload.use_finetuned:
        if FINETUNED_MODEL is None:
            raise HTTPException(status_code=400, detail="Fine-tuned model not available.")
        model = FINETUNED_MODEL
        model_label = "fine-tuned"

    try:
        summary = model.summarize(payload.text, generation=generation)
        return SummarizeResponse(summary=summary, model_used=model_label)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)