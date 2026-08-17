"""FastAPI inference server: Load champion adapter, serve predictions, auto-reload."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

log = structlog.get_logger(__name__)

# Load environment
load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./outputs"))
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
MODEL_RELOAD_INTERVAL = int(os.getenv("PIPELINE_MODEL_RELOAD_INTERVAL", "60"))


class PredictionRequest(BaseModel):
    """Inference request."""

    prompt: str
    max_tokens: int = 256


class PredictionResponse(BaseModel):
    """Inference response."""

    prompt: str
    prediction: str
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_version: Optional[str]
    last_reload: Optional[str]
    adapter_path: Optional[str]


class InferenceServer:
    """Load and serve DPO-trained adapter."""

    def __init__(self):
        self.adapter_path: Optional[Path] = None
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_version: Optional[str] = None
        self.last_reload: Optional[datetime] = None
        self.champion_pointer_path = OUTPUT_DIR / "champion_pointer.json"

        log.info(f"InferenceServer initializing")
        log.info(f"  Device: {self.device}")
        log.info(f"  Output dir: {OUTPUT_DIR}")

        # Load initial model
        self._load_model()

    def _get_champion_adapter_path(self) -> Optional[Path]:
        """Get champion adapter path from pointer file or default."""
        # Check for champion pointer
        if self.champion_pointer_path.exists():
            try:
                with open(self.champion_pointer_path) as f:
                    pointer = json.load(f)
                    log.info(f"Champion pointer found: {pointer.get('job_id')}")
                    # Return the outputs/latest directory
                    latest_path = OUTPUT_DIR / "latest"
                    if latest_path.exists():
                        return latest_path
            except Exception as e:
                log.warning(f"Failed to read champion pointer: {str(e)}")

        # Fallback to outputs/latest or outputs/checkpoint-1973
        fallback_paths = [
            OUTPUT_DIR / "latest",
            OUTPUT_DIR / "checkpoint-1973",
            OUTPUT_DIR,
        ]

        for path in fallback_paths:
            adapter_config = path / "adapter_config.json"
            if adapter_config.exists():
                return path

        return None

    def _load_model(self) -> None:
        """Load base model + LoRA adapter."""
        try:
            # Get champion adapter path
            adapter_path = self._get_champion_adapter_path()

            if not adapter_path or not (adapter_path / "adapter_config.json").exists():
                log.warning(f"No trained adapter found. Using base model only.")
                self.adapter_path = None
                self._load_base_model_only()
                return

            log.info(f"Loading adapter from {adapter_path}")

            # Load base model
            log.info(f"Loading base model: {BASE_MODEL}")
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.model.eval()

            self.adapter_path = adapter_path
            self.model_version = adapter_path.name
            self.last_reload = datetime.now()

            log.info(f"✓ Model loaded with adapter: {self.model_version}")

        except Exception as e:
            log.error(f"Failed to load model: {str(e)}")
            log.warning(f"Falling back to base model")
            self._load_base_model_only()

    def _load_base_model_only(self) -> None:
        """Load base model without adapter."""
        try:
            log.info(f"Loading base model only: {BASE_MODEL}")
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()

            self.adapter_path = None
            self.model_version = "base"
            self.last_reload = datetime.now()

            log.info(f"✓ Base model loaded")
        except Exception as e:
            log.error(f"Failed to load base model: {str(e)}")
            self.model = None
            self.tokenizer = None

    def reload_if_needed(self) -> None:
        """Check for new champion and reload if available."""
        if not self.champion_pointer_path.exists():
            return

        try:
            with open(self.champion_pointer_path) as f:
                pointer = json.load(f)
                job_id = pointer.get("job_id")

                # If job_id changed, reload
                if job_id and job_id != self.model_version:
                    log.info(f"New champion detected: {job_id}")
                    self._load_model()
        except Exception as e:
            log.debug(f"Error checking for new champion: {str(e)}")

    def predict(self, prompt: str, max_tokens: int = 256) -> str:
        """Run inference."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")

        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True,
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            # Decode
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return prediction

        except Exception as e:
            log.error(f"Prediction failed: {str(e)}")
            raise


# Initialize server
server = InferenceServer()

# Create FastAPI app
app = FastAPI(
    title="DPO Inference Server",
    description="Serves DPO-trained LLM adapter",
    version="1.0.0",
)


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    log.info("FastAPI server starting")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    log.info("FastAPI server shutting down")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    # Check for new champion
    server.reload_if_needed()

    return HealthResponse(
        status="healthy" if server.model is not None else "unhealthy",
        model_loaded=server.model is not None,
        model_version=server.model_version,
        last_reload=server.last_reload.isoformat() if server.last_reload else None,
        adapter_path=str(server.adapter_path) if server.adapter_path else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Inference endpoint."""
    if server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        prediction = server.predict(request.prompt, request.max_tokens)
        return PredictionResponse(
            prompt=request.prompt,
            prediction=prediction,
            model_version=server.model_version or "unknown",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        log.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "DPO Inference Server",
        "version": "1.0.0",
        "status": "running",
        "model": server.model_version,
    }


if __name__ == "__main__":
    import uvicorn

    try:
        from src.infra import setup_logging

        setup_logging(environment="production")
    except ImportError:
        pass

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
