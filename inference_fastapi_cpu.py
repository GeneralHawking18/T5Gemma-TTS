"""
T5Gemma-TTS ONNX-Accelerated FastAPI Service.

Usage:
    python inference_fastapi_cpu.py
"""

import os
import gc
import time
import io
import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Local imports
from onnx_inference import T5GemmaONNX
from transformers import AutoTokenizer

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
MODEL_DIR = "./onnx_models_int8"  # Point to INT8 directory
USE_INT8 = True
DEVICE = "cpu"

class TTSState:
    engine: Optional[T5GemmaONNX] = None
    tokenizer = None

state = TTSState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX models on startup."""
    logger.info(f"Loading ONNX models from {MODEL_DIR}...")
    start_time = time.time()
    
    # 1. Init Engine
    state.engine = T5GemmaONNX(MODEL_DIR, use_int8=USE_INT8, device=DEVICE)
    
    # 2. Init Tokenizer (Text)
    # Ideally load from local or config. Using hardcoded or config-based.
    # Should get name from model_args.json via engine.config
    tokenizer_name = state.engine.config.get("text_tokenizer_name", "google/t5gemma-b-b-ul2")
    state.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    logger.info(f"Ready in {time.time() - start_time:.2f}s")
    yield
    
    # Cleaning
    del state.engine
    gc.collect()

app = FastAPI(title="T5Gemma ONNX TTS", lifespan=lifespan)

class SynthesisRequest(BaseModel):
    text: str
    speed: float = 1.0 # Placeholder
    target_len: int = 250 # Max tokens

@app.get("/health")
def health():
    return {"status": "ok", "backend": "onnx", "int8": USE_INT8}

@app.post("/synthesize")
async def synthesize(req: SynthesisRequest):
    if not state.engine:
        raise HTTPException(503, "Engine not loaded")
        
    try:
        # 1. Tokenize Text
        text_tokens = state.tokenizer.encode(req.text, return_tensors="numpy")
        
        # 2. Generate Codes
        # Approximate length: 50 tokens ~ 1 sec? (depends on codec)
        # Using simple length heuristic if not provided
        codes, dur = state.engine.generate(
            text_tokens=text_tokens,
            target_len=req.target_len
        )
        
        # 3. Decode Audio
        audio_wav = state.engine.decode_audio(codes)
        
        # 4. To WAV Base64
        buffer = io.BytesIO()
        sf.write(buffer, audio_wav, 16000, format='WAV') # Assuming 16k
        buffer.seek(0)
        b64_audio = base64.b64encode(buffer.read()).decode()
        
        return {
            "text": req.text,
            "inference_time": dur,
            "audio_base64": b64_audio
        }
        
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
