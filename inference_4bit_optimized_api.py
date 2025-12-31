"""
Optimized T5Gemma-TTS REST API Server (4-bit + torch.compile)

Features:
- 4-bit Quantization (bitsandbytes) for low VRAM usage.
- torch.compile (JIT) for accelerated autoregressive decoding.
- TF32 precision enabled for Ampere+ GPUs.
- Thread-safe GPU access (Async Lock).
- Configurable via Environment Variables.

Usage:
    python inference_4bit_optimized_api.py
"""

import io
import os
import base64
import time
import random
import logging
import asyncio
import traceback
import numpy as np
import scipy.io.wavfile as wav
import torch
import uvicorn

from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import (
    inference_one_sample,
    normalize_text_with_lang,
)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    raise ImportError(
        "transformers is not installed. Please install it with: "
        "pip install transformers bitsandbytes accelerate"
    )

# ============================================================================
# Configuration & Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("T5Gemma-API")

# Enable TF32 for faster FP32 math on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    logger.info("TF32 optimization enabled.")
    
    # Add CUDA to PATH if not present (Fix for torch.compile)
    cuda_path = "/usr/local/cuda-12.8"
    if os.path.exists(cuda_path):
        os.environ["PATH"] = f"{cuda_path}/bin:" + os.environ.get("PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
        logger.info(f"Added {cuda_path} to PATH/LD_LIBRARY_PATH")

# ============================================================================
# Pydantic Models
# ============================================================================

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", min_length=1)
    language: Optional[str] = Field(None, description="Language code (e.g., 'ja', 'en').")
    target_duration: Optional[float] = Field(None, ge=0.1, le=60.0, description="Duration in seconds.")
    top_k: int = Field(30, ge=1, le=200, description="Top-k sampling")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    seed: Optional[int] = Field(None, description="Random seed")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speaking speed multiplier (approximate)")


class SynthesizeResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    inference_time: float


class ModelInfo(BaseModel):
    model_name: str
    quantization: str
    device: str
    sample_rate: int
    compiled: bool
    status: str


# ============================================================================
# TTS Engine
# ============================================================================

class TTSEngine:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.config = None
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.lock = asyncio.Lock()  # Ensure serial access to GPU
        self.is_loaded = False
        self.is_compiled = False

    def load(self):
        if self.device == "cpu":
            raise RuntimeError("CUDA is required for 4-bit quantization.")

        logger.info(f"Loading 4-bit model from: {self.model_dir}")
        
        # Check for Flash Attention 2
        attn_impl = "eager"
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            logger.info("Flash Attention 2 enabled.")
        except ImportError:
            logger.warning("flash_attn not found, using default attention.")

        # Load Model (4-bit + bfloat16)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        self.model.eval()
        self.config = self.model.config

        # Tokenizers
        tokenizer_name = getattr(self.config, "text_tokenizer_name", None) or \
                         getattr(self.config, "t5gemma_model_name", None)
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.audio_tokenizer = AudioTokenizer(
            backend="xcodec2",
            model_name=getattr(self.config, "xcodec2_model_name", "xcodec2"),
        )

        # Optimize: torch.compile
        # Stability decision: Disabled by default. Flash Attention provides consistent speedups.
        # Set ENABLE_COMPILE=1 to experiment with compilation.
        if hasattr(torch, "compile") and os.environ.get("ENABLE_COMPILE", "0") == "1":
            self._apply_compilation()
        
        self.is_loaded = True
        logger.info("Model loaded successfully.")

    def _apply_compilation(self):
        logger.info("Applying torch.compile to Decoder Module...")
        try:
            target_module = None
            if hasattr(self.model, "decoder_module"):
                target_module = self.model.decoder_module
            elif hasattr(self.model, "model") and hasattr(self.model.model, "decoder"):
                target_module = self.model.model.decoder
            
            if target_module:
                # 'max-autotune-no-cudagraphs' optimizes kernels heavily but avoids CUDAGraphs memory issues
                self.model.decoder_module = torch.compile(
                    target_module, 
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=False
                )
                self.is_compiled = True
                logger.info("Compilation enabled (lazy, max-autotune-no-cudagraphs).")
            else:
                logger.warning("Could not find decoder module to compile.")
        except Exception as e:
            logger.error(f"torch.compile failed: {e}")
            self.is_compiled = False

    def warmup(self):
        """Trigger JIT compilation with a dummy run."""
        if not self.is_loaded:
            return
        
        logger.info("Running warmup inference...")
        try:
            self._synthesize_internal(
                text="Warmup",
                target_duration=1.0,
                seed=42
            )
            logger.info("Warmup complete.")
        except Exception as e:
            logger.error(f"Warmup failed: {e}")

    def _seed_everything(self, seed: int):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _synthesize_internal(self, text, **kwargs) -> tuple:
        """Internal synchronous synthesis function."""
        # Unpack args
        seed = kwargs.get('seed')
        if seed is not None:
            self._seed_everything(seed)
        
        lang = kwargs.get('language')
        lang = None if lang in {None, "", "none"} else str(lang)
        
        target_text, lang_code = normalize_text_with_lang(text, lang)
        
        # Duration estimation
        target_duration = kwargs.get('target_duration')
        if target_duration is None:
            est = estimate_duration(target_text, target_lang=lang_code, reference_lang=lang_code)
            # Apply speed modifier if needed (simplified)
            speed = kwargs.get('speed', 1.0)
            target_duration = est / speed
        else:
            target_duration = float(target_duration)

        codec_audio_sr = self.audio_tokenizer.sample_rate
        codec_sr = getattr(self.config, "encodec_sr", 50)

        decode_config = {
            "top_k": kwargs.get('top_k', 30),
            "top_p": kwargs.get('top_p', 0.9),
            "min_p": 0,
            "temperature": kwargs.get('temperature', 0.7),
            "stop_repetition": 3,
            "codec_audio_sr": codec_audio_sr,
            "codec_sr": codec_sr,
            "silence_tokens": [],
            "sample_batch_size": 1,
        }

        with torch.inference_mode():
            concat_audio, gen_audio = inference_one_sample(
                model=self.model,
                model_args=self.config,
                text_tokenizer=self.text_tokenizer,
                audio_tokenizer=self.audio_tokenizer,
                audio_fn=None,
                target_text=target_text,
                lang=lang_code,
                device=self.device,
                decode_config=decode_config,
                prompt_end_frame=0,
                target_generation_length=target_duration,
                prefix_transcript="",
                multi_trial=[],
                repeat_prompt=0,
                return_frames=False,
            )

        gen_audio_np = gen_audio[0].cpu().numpy()
        if gen_audio_np.ndim > 1:
            gen_audio_np = gen_audio_np.flatten()
            
        return codec_audio_sr, gen_audio_np

    async def synthesize(self, request: SynthesizeRequest):
        """Async wrapper with Lock for thread safety."""
        async with self.lock:
            # Offload to thread if needed, but inference_one_sample is GPU bound.
            # Running directly in async loop blocks it, but since we have a lock
            # and it's single-GPU, blocking is effectively the same as serializing.
            # To be 100% proper, we could use run_in_executor, but torch cuda context
            # management across threads can be tricky. Direct call is safer here.
            
            return self._synthesize_internal(
                text=request.text,
                language=request.language,
                target_duration=request.target_duration,
                top_k=request.top_k,
                top_p=request.top_p,
                temperature=request.temperature,
                seed=request.seed,
                speed=request.speed
            )

# ============================================================================
# Application Setup
# ============================================================================

engine: Optional[TTSEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    
    model_dir = os.environ.get(
        "T5GEMMA_MODEL_DIR", 
        "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit"
    )
    
    engine = TTSEngine(model_dir)
    try:
        engine.load()
        engine.warmup()
    except Exception as e:
        logger.critical(f"Fatal error loading model: {e}")
        traceback.print_exc()
        # We don't exit here to allow /health to report failure
    
    yield
    
    # Cleanup
    if engine and engine.model:
        del engine.model
        torch.cuda.empty_cache()
        logger.info("Shutdown complete.")

app = FastAPI(title="T5Gemma-TTS Optimized API", version="2.0", lifespan=lifespan)

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    ready = engine is not None and engine.is_loaded
    return {
        "status": "ready" if ready else "not_ready",
        "device": engine.device if engine else None,
        "compiled": engine.is_compiled if engine else False
    }

@app.get("/model", response_model=ModelInfo)
async def model_info():
    if not engine or not engine.is_loaded:
        raise HTTPException(503, "Model not loaded")
    return ModelInfo(
        model_name=engine.model_dir,
        quantization="4-bit",
        device=engine.device,
        sample_rate=engine.audio_tokenizer.sample_rate,
        compiled=engine.is_compiled,
        status="active"
    )

@app.post("/synthesize")
async def synthesize_endpoint(req: SynthesizeRequest):
    if not engine or not engine.is_loaded:
        raise HTTPException(503, "Model loading...")
    
    try:
        sample_rate, audio = await engine.synthesize(req)
        
        # Format audio
        if audio.dtype != np.int16:
            max_val = np.abs(audio).max()
            if max_val > 0: audio = audio / max_val
            audio = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        wav.write(buffer, sample_rate, audio)
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Synthesis failed: {str(e)}")

@app.post("/synthesize_base64", response_model=SynthesizeResponse)
async def synthesize_base64_endpoint(req: SynthesizeRequest):
    if not engine or not engine.is_loaded:
        raise HTTPException(503, "Model loading...")

    try:
        t0 = time.time()
        sample_rate, audio = await engine.synthesize(req)
        dur = time.time() - t0
        
        # Format audio
        if audio.dtype != np.int16:
            max_val = np.abs(audio).max()
            if max_val > 0: audio = audio / max_val
            audio = (audio * 32767).astype(np.int16)

        buffer = io.BytesIO()
        wav.write(buffer, sample_rate, audio)
        buffer.seek(0)
        
        b64_data = base64.b64encode(buffer.read()).decode('utf-8')
        
        return SynthesizeResponse(
            audio_base64=b64_data,
            sample_rate=sample_rate,
            inference_time=dur
        )
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("inference_4bit_optimized_api:app", host=host, port=port)
