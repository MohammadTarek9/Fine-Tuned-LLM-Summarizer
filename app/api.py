"""FastAPI starter for summarization inference."""
from __future__ import annotations
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from src.model import DomainSummarizer, GenerationParams
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.model import DomainSummarizer, GenerationParams

BASE_MODEL = DomainSummarizer(model_name = "google/flan-t5-base")
FINETUNED_MODEL: Optional[DomainSummarizer] = None

class SummarizeRequest(BaseModel):
    """input schema for summarize endpoint"""
    text: str = Field(..., min_length=10, description="Article text to summarize.")
    temperature: float = Field(0.7, ge=0.01, le=2.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.95, ge=0.1, le=1.0)
    max_new_tokens: int = Field(128, ge=16, le=5000)
    use_finetuned: bool = Field(False, description="Use LoRA adapter if available.")

class SummarizeResponse(BaseModel):
    """output schema for summarize endpoint"""
    summary: str
    model_used: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """load models at startup"""
    try:
        BASE_MODEL.load()
        app.state.base_model = BASE_MODEL
    except Exception as exc:
        raise RuntimeError("Failed to load base summarizer on startup.") from exc

    adapter_dir = Path("./lora-adapter")
    finetuned_model = None

    if adapter_dir.exists():
        try:
            finetuned_model = DomainSummarizer(
                model_name="google/flan-t5-base",
                adapter_path=str(adapter_dir),
            )
            finetuned_model.load() 
        except Exception:
            finetuned_model = None

    app.state.finetuned_model = finetuned_model

    yield

app = FastAPI(title="Domain Summarizer API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "base_model_loaded": app.state.base_model.model is not None,
        "finetuned_model_loaded": (
            app.state.finetuned_model is not None
            and app.state.finetuned_model.model is not None
        ),
    }

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest):
    generation = GenerationParams(
        temperature=payload.temperature,
        top_k=payload.top_k,
        top_p=payload.top_p,
        max_new_tokens=payload.max_new_tokens
    )

    model = app.state.base_model
    model_label = "base"

    if payload.use_finetuned:
        if app.state.finetuned_model is None:
            raise HTTPException(status_code=400, detail="Fine-tuned model not available.")
        model = app.state.finetuned_model
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