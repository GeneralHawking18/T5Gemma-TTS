#!/usr/bin/env python3
"""
T5Gemma-TTS FastAPI Server.

Load pre-quantized 4-bit model từ HuggingFace cache.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import gc
import io
import time
import base64
from typing import Optional
from contextlib import asynccontextmanager
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
torch.set_grad_enabled(False)
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # Already set in parent process during reload

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit"

# ============================================================
# GLOBALS
# ============================================================
model = None
text_tokenizer = None
audio_tokenizer = None
model_config = None


# ============================================================
# PYDANTIC MODELS
# ============================================================
class SynthesizeRequest(BaseModel):
    text: str = "ただいま確認しておりますので、少々お待ちください。"
    target_duration: Optional[float] = None
    top_k: int = 30
    top_p: float = 0.9
    temperature: float = 0.8
    lang: Optional[str] = None


class SynthesizeResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    duration: float
    inference_time: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ============================================================
# MODEL LOADING
# ============================================================
def load_model():
    """Load model trực tiếp từ HuggingFace cache."""
    global model, text_tokenizer, audio_tokenizer, model_config
    
    print(f"[Load] Loading model from {MODEL_ID}...")
    
    print(f"[Load] Loading model from {MODEL_ID}...")
    
    # Use Hybrid ONNX-PyTorch Model
    try:
        from inference_hybrid import HybridT5Gemma
        
        has_cuda = torch.cuda.is_available()
        device = "cuda" if has_cuda else "cpu"
        
        print(f"[Load] Device: {device}")
        
        model = HybridT5Gemma(
            model_name=MODEL_ID,
            onnx_dir="./onnx_models_int8",
            device=device,
            use_int8=True,
        )
    except Exception as e:
        print(f"[Error] Failed to load HybridT5Gemma: {e}")
        traceback.print_exc()
        raise e

    model_config = model.model.config # HybridT5Gemma wraps the HF model in .model
    
    print("[OK] Model loaded!")
    
    # Tokenizer
    tokenizer_name = (
        getattr(model_config, "text_tokenizer_name", None) or
        getattr(model_config, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2")
    )
    print(f"[Load] Loading tokenizer: {tokenizer_name}")
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Audio tokenizer
    print("[Load] Loading XCodec2...")
    from data.tokenizer import AudioTokenizer
    
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(model_config, "xcodec2_model_name", None),
        device="cpu",
    )
    
    gc.collect()
    print("[Load] Ready!")


# ============================================================
# LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    global model, text_tokenizer, audio_tokenizer
    del model, text_tokenizer, audio_tokenizer
    gc.collect()


# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="T5Gemma-TTS API",
    description="TTS inference (4-bit pre-quantized)",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    print("[Synthesize] Synthesizing...")
    global model, text_tokenizer, audio_tokenizer, model_config
    
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    try:
        from inference_tts_utils import inference_one_sample, normalize_text_with_lang
        from duration_estimator import estimate_duration
        
        start_time = time.time()
        
        target_text, lang_code = normalize_text_with_lang(request.text, request.lang)
        
        if request.target_duration is None:
            target_generation_length = estimate_duration(
                target_text=target_text,
                reference_speech=None,
                reference_transcript=None,
                target_lang=lang_code,
                reference_lang=lang_code,
            )
        else:
            target_generation_length = request.target_duration
        
        codec_audio_sr = audio_tokenizer.sample_rate
        codec_sr = getattr(model_config, "encodec_sr", 50)
        
        decode_config = {
            'top_k': request.top_k,
            'top_p': request.top_p,
            'min_p': 0,
            'temperature': request.temperature,
            'stop_repetition': 3,
            'codec_audio_sr': codec_audio_sr,
            'codec_sr': codec_sr,
            'silence_tokens': [],
            'sample_batch_size': 1,
        }
        
        with torch.inference_mode():
            _, gen_audio = inference_one_sample(
                model=model,
                model_args=model_config,
                text_tokenizer=text_tokenizer,
                audio_tokenizer=audio_tokenizer,
                audio_fn=None,
                target_text=target_text,
                lang=lang_code,
                device=model.device,
                decode_config=decode_config,
                prompt_end_frame=0,
                target_generation_length=target_generation_length,
                prefix_transcript="",
                multi_trial=[],
                repeat_prompt=0,
                return_frames=False,
            )
        
        gen_audio = gen_audio[0].cpu().numpy()
        inference_time = time.time() - start_time
        duration = len(gen_audio) / codec_audio_sr
        
        from scipy.io import wavfile
        buffer = io.BytesIO()
        # Convert float audio to int16 for WAV (normalize and clip to prevent overflow)
        if gen_audio.dtype in (np.float32, np.float64):
            gen_audio = np.clip(gen_audio, -1.0, 1.0)
            gen_audio = (gen_audio * 32767).astype(np.int16)
        wavfile.write(buffer, codec_audio_sr, gen_audio)
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode()
        
        gc.collect()
        
        return SynthesizeResponse(
            audio_base64=audio_base64,
            sample_rate=codec_audio_sr,
            duration=duration,
            inference_time=inference_time,
        )
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/synthesize/wav")
async def synthesize_wav(request: SynthesizeRequest):
    result = await synthesize(request)
    audio_bytes = base64.b64decode(result.audio_base64)
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=output.wav",
            "X-Duration": str(result.duration),
            "X-Inference-Time": str(result.inference_time),
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
