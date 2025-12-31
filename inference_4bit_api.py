"""
T5Gemma-TTS REST API Server for TTS Inference using 4-bit quantization (GPU)
This API loads the model in 4-bit using bitsandbytes for memory efficiency
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

from typing import Optional, List
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
        "transformers is not installed. Please install it with "
        "`pip install transformers bitsandbytes accelerate`."
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("T5Gemma-TTS-API")


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
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class SynthesizeResponse(BaseModel):
    """Response for /synthesize_base64 endpoint"""
    audio_base64: str
    sample_rate: int
    inference_time: float


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    quantization: str
    device: str
    sample_rate: int


# ============================================================================
# Global State
# ============================================================================

class TTSModelHolder:
    """Holds the loaded TTS model and its components"""
    
    def __init__(self):
        self.model = None
        self.config = None
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.device = None
        self.codec_audio_sr = 16000
        self.codec_sr = 50
        self.model_dir = None
        self.is_loaded = False
    
    def seed_everything(self, seed: int):
        """Set random seeds for reproducibility"""
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    def load_model(self, model_dir: str = "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit"):
        """Load the 4-bit quantized model"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. 4-bit quantization requires a GPU.")
        
        self.device = "cuda"
        self.model_dir = model_dir
        
        logger.info(f"Loading pre-quantized 4-bit model from {model_dir}...")
        
        # Load pre-quantized 4-bit model with bfloat16 compute dtype
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.config = self.model.config
        
        # Load text tokenizer
        tokenizer_name = getattr(self.config, "text_tokenizer_name", None) or \
                         getattr(self.config, "t5gemma_model_name", None)
        self.text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Text tokenizer loaded: {tokenizer_name}")
        
        # Load audio tokenizer (xcodec2)
        self.audio_tokenizer = AudioTokenizer(
            backend="xcodec2",
            model_name=getattr(self.config, "xcodec2_model_name", "xcodec2"),
        )
        
        # Update sample rates from config
        self.codec_audio_sr = getattr(self.config, "codec_audio_sr", self.codec_audio_sr)
        self.codec_sr = getattr(self.config, "encodec_sr", self.codec_sr)
        # Align with audio tokenizer sample rate
        self.codec_audio_sr = self.audio_tokenizer.sample_rate
        
        self.is_loaded = True
        logger.info(f"Model loaded successfully. Sample rate: {self.codec_audio_sr}")
    
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        target_duration: Optional[float] = None,
        top_k: int = 30,
        top_p: float = 0.9,
        min_p: float = 0,
        temperature: float = 0.7,
        stop_repetition: int = 3,
        sample_batch_size: int = 1,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Run TTS inference and return (sample_rate, audio_numpy)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Set seed if provided
        if seed is not None:
            self.seed_everything(seed)
        
        # Normalize text and detect language
        lang = None if language in {None, "", "none", "null"} else str(language)
        target_text, lang_code = normalize_text_with_lang(text, lang)
        
        # Estimate duration if not provided
        if target_duration is None:
            target_generation_length = estimate_duration(
                target_text=target_text,
                reference_speech=None,
                reference_transcript=None,
                target_lang=lang_code,
                reference_lang=lang_code,
            )
            logger.info(f"Estimated duration: {target_generation_length:.2f}s")
        else:
            target_generation_length = float(target_duration)
        
        decode_config = {
            "top_k": top_k,
            "top_p": top_p,
            "min_p": min_p,
            "temperature": temperature,
            "stop_repetition": stop_repetition,
            "codec_audio_sr": self.codec_audio_sr,
            "codec_sr": self.codec_sr,
            "silence_tokens": [],
            "sample_batch_size": sample_batch_size,
        }
        
        # Run inference
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
            target_generation_length=target_generation_length,
            prefix_transcript="",
            multi_trial=[],
            repeat_prompt=0,
            return_frames=False,
        )
        
        # Convert to numpy and ensure 1D shape
        gen_audio = gen_audio[0].cpu().numpy()
        # Flatten if multi-dimensional (e.g., (1, N) -> (N,))
        if gen_audio.ndim > 1:
            gen_audio = gen_audio.flatten()
        
        return self.codec_audio_sr, gen_audio
    
    def unload(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.audio_tokenizer is not None:
            del self.audio_tokenizer
            self.audio_tokenizer = None
        torch.cuda.empty_cache()
        self.is_loaded = False
        logger.info("Model unloaded")


# Global model holder
model_holder: Optional[TTSModelHolder] = None


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global model_holder
    
    model_holder = TTSModelHolder()
    
    # Get model path from environment or use default
    model_dir = os.environ.get(
        "T5GEMMA_MODEL_DIR",
        "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit"
    )
    
    try:
        model_holder.load_model(model_dir)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
    
    yield
    
    # Cleanup
    if model_holder:
        model_holder.unload()


app = FastAPI(
    title="T5Gemma-TTS 4-bit API",
    description="REST API for T5Gemma Text-to-Speech synthesis using 4-bit quantization",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": model_holder is not None and model_holder.is_loaded,
        "runtime": "pytorch-4bit",
        "device": model_holder.device if model_holder else None,
    }


@app.get("/model", response_model=ModelInfo)
async def get_model_info():
    """Get loaded model information"""
    if model_holder is None or not model_holder.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name=model_holder.model_dir or "unknown",
        quantization="4-bit",
        device=model_holder.device,
        sample_rate=model_holder.codec_audio_sr,
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
            temperature=request.temperature,
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
            temperature=request.temperature,
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
    uvicorn.run("inference_4bit_api:app", host="0.0.0.0", port=8000, reload=True)
