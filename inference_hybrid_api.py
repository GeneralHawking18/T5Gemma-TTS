"""
T5Gemma-TTS REST API Server for Hybrid ONNX+PyTorch Inference
This API uses:
- ONNX Encoder (fast, CPU-friendly)
- PyTorch Decoder (T5GemmaVoiceModel with PM-RoPE)
- XCodec2 Audio Tokenizer

Reuses code from:
- inference_hybrid_complete.py: HybridT5GemmaTTS class with synthesize() method
"""
import io
import os
import base64
import traceback
import time
import random
import logging

import numpy as np
import torch
import scipy.io.wavfile as wav

from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from inference_hybrid_complete import HybridT5GemmaTTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("T5Gemma-TTS-Hybrid-API")


# ============================================================================
# Pydantic Models
# ============================================================================

class SynthesizeRequest(BaseModel):
    """Request body for /synthesize endpoint"""
    text: str = Field(..., description="Text to synthesize")
    language: Optional[str] = Field(None, description="Language code (e.g., 'ja', 'en'). Auto-detect if not specified.")
    target_duration: Optional[float] = Field(None, ge=0.1, le=60.0, description="Target duration in seconds. Auto-estimate if not specified.")
    top_k: int = Field(30, ge=1, le=100, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p (nucleus) sampling")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Min-p sampling")
    stop_repetition: int = Field(3, ge=1, le=10, description="Stop repetition threshold")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SynthesizeResponse(BaseModel):
    """Response for /synthesize_base64 endpoint"""
    audio_base64: str
    sample_rate: int
    inference_time: float


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    encoder_backend: str
    decoder_backend: str
    device: str
    sample_rate: int


# ============================================================================
# Global State
# ============================================================================

class HybridTTSModelHolder:
    """Holds the loaded Hybrid TTS model (reuses HybridT5GemmaTTS)"""

    def __init__(self):
        self.tts = None
        self.device = None
        self.onnx_dir = None
        self.decoder_weights = None
        self.is_loaded = False

    def seed_everything(self, seed: int):
        """Set random seeds for reproducibility"""
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def load_model(
        self,
        onnx_dir: str = "onnx_models_fp16_fixed",
        decoder_weights: str = "weights/decoder_pmrope.bin",
    ):
        """Load the hybrid ONNX+PyTorch model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.onnx_dir = onnx_dir
        self.decoder_weights = decoder_weights

        logger.info(f"Loading Hybrid TTS model (ONNX encoder + PyTorch decoder)...")
        logger.info(f"  - ONNX dir: {onnx_dir}")
        logger.info(f"  - Decoder weights: {decoder_weights}")
        logger.info(f"  - Device: {self.device}")

        # Load hybrid TTS model (includes audio tokenizer via lazy loading)
        self.tts = HybridT5GemmaTTS(
            onnx_dir=onnx_dir,
            decoder_weights=decoder_weights,
            device=self.device,
        )

        self.is_loaded = True
        logger.info(f"Hybrid TTS model loaded successfully!")

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        target_duration: Optional[float] = None,
        top_k: int = 30,
        top_p: float = 0.9,
        min_p: float = 0.0,
        temperature: float = 0.7,
        stop_repetition: int = 3,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Run TTS inference and return (sample_rate, audio_numpy).
        Delegates to HybridT5GemmaTTS.synthesize() method.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Set seed if provided
        if seed is not None:
            self.seed_everything(seed)

        # Delegate to the HybridT5GemmaTTS.synthesize() method
        return self.tts.synthesize(
            text=text,
            language=language,
            target_duration=target_duration,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            stop_repetition=stop_repetition,
        )

    def unload(self):
        """Unload model to free memory"""
        if self.tts is not None:
            # Clean up audio tokenizer if loaded
            if hasattr(self.tts, '_audio_tokenizer') and self.tts._audio_tokenizer is not None:
                del self.tts._audio_tokenizer
            del self.tts
            self.tts = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("Model unloaded")


# Global model holder
model_holder: Optional[HybridTTSModelHolder] = None


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global model_holder

    model_holder = HybridTTSModelHolder()

    # Get model paths from environment or use defaults
    onnx_dir = os.environ.get("ONNX_DIR", "onnx_models_fp16_fixed")
    decoder_weights = os.environ.get("DECODER_WEIGHTS", "weights/decoder_pmrope.bin")

    try:
        model_holder.load_model(
            onnx_dir=onnx_dir,
            decoder_weights=decoder_weights,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()

    yield

    # Cleanup
    if model_holder:
        model_holder.unload()


app = FastAPI(
    title="T5Gemma-TTS Hybrid API",
    description="REST API for T5Gemma Text-to-Speech synthesis using Hybrid ONNX+PyTorch",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": model_holder is not None and model_holder.is_loaded,
        "runtime": "hybrid-onnx-pytorch",
        "device": model_holder.device if model_holder else None,
    }


@app.get("/model", response_model=ModelInfo)
async def get_model_info():
    """Get loaded model information"""
    if model_holder is None or not model_holder.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get sample rate (load audio tokenizer if needed)
    audio_tokenizer = model_holder.tts._load_audio_tokenizer()
    sample_rate = audio_tokenizer.sample_rate

    return ModelInfo(
        model_name="T5Gemma-TTS-2b-2b",
        encoder_backend="ONNX (CPU)",
        decoder_backend=f"PyTorch ({model_holder.device})",
        device=model_holder.device,
        sample_rate=sample_rate,
    )


@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech from text.
    Returns WAV audio file.
    """
    if model_holder is None or not model_holder.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        sample_rate, audio = model_holder.synthesize(
            text=request.text,
            language=request.language,
            target_duration=request.target_duration,
            top_k=request.top_k,
            top_p=request.top_p,
            min_p=request.min_p,
            temperature=request.temperature,
            stop_repetition=request.stop_repetition,
            seed=request.seed,
        )

        # Convert to WAV bytes
        wav_buffer = io.BytesIO()

        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()

        # Ensure audio is in correct format for scipy (int16)
        if audio.dtype != np.int16:
            # Normalize to [-1, 1] then convert to int16
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio_normalized = audio / max_val
            else:
                audio_normalized = audio
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
        else:
            audio_int16 = audio

        wav.write(wav_buffer, int(sample_rate), audio_int16)
        wav_buffer.seek(0)

        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_base64", response_model=SynthesizeResponse)
async def synthesize_base64(request: SynthesizeRequest):
    """
    Synthesize speech from text.
    Returns base64 encoded WAV audio.
    """
    if model_holder is None or not model_holder.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        time_start = time.time()

        sample_rate, audio = model_holder.synthesize(
            text=request.text,
            language=request.language,
            target_duration=request.target_duration,
            top_k=request.top_k,
            top_p=request.top_p,
            min_p=request.min_p,
            temperature=request.temperature,
            stop_repetition=request.stop_repetition,
            seed=request.seed,
        )

        inference_time = time.time() - time_start

        # Convert to WAV bytes
        wav_buffer = io.BytesIO()

        # Ensure audio is 1D
        if audio.ndim > 1:
            audio = audio.flatten()

        # Ensure audio is in correct format for scipy (int16)
        if audio.dtype != np.int16:
            # Normalize to [-1, 1] then convert to int16
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio_normalized = audio / max_val
            else:
                audio_normalized = audio
            audio_int16 = (audio_normalized * 32767).astype(np.int16)
        else:
            audio_int16 = audio

        wav.write(wav_buffer, int(sample_rate), audio_int16)
        wav_buffer.seek(0)

        audio_base64 = base64.b64encode(wav_buffer.read()).decode("utf-8")

        return SynthesizeResponse(
            audio_base64=audio_base64,
            sample_rate=sample_rate,
            inference_time=inference_time,
        )

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_hybrid_api:app", host="0.0.0.0", port=8000, reload=True)
