"""
T5Gemma-TTS REST API Server for TensorRT Inference
This API uses:
- ONNX Encoder (fast, CPU-friendly)
- TensorRT Decoder (Ultra-fast GPU execution)
- XCodec2 Audio Tokenizer
"""
import io
import os
import json
import base64
import traceback
import time
import random
import logging

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch
import tensorrt as trt
import tensorrt_llm
import scipy.io.wavfile as wav
import onnxruntime as ort
from transformers import AutoTokenizer

from typing import Optional, List, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("T5Gemma-TTS-TRT-API")

# Add current dir to path for local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.utils import topk_sampling

# ============================================================================
# Pydantic Models
# ============================================================================

class SynthesizeRequest(BaseModel):
    """Request body for /synthesize endpoint"""
    text: str = Field(..., description="Text to synthesize")
    language: Optional[str] = Field(None, description="Language code (e.g., 'ja', 'en'). Auto-detect if not specified.")
    target_duration: Optional[float] = Field(None, ge=0.1, le=60.0, description="Target duration in seconds.")
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
# Core TRT Engine Wrapper
# ============================================================================

class TRTDecoderWrapper:
    """Wrapper for TensorRT-LLM Decoder Engine"""
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        # Register plugins
        trt.init_libnvinfer_plugins(self.logger, "")
        
        logger.info(f"Loading TRT Decoder engine: {engine_path}")
        with open(engine_path, "rb") as f:
            engine_buffer = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_buffer)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def run(self, inputs: dict):
        """Execute TRT engine with given inputs (dict of torch tensors)"""
        # Set input shapes for dynamic dimensions
        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tensor.shape)
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Allocate output buffer
        output_name = "output"
        output_shape = self.context.get_tensor_shape(output_name)
        # Handle dynamic output shape if necessary (though usually fixed by input shapes)
        output_tensor = torch.empty(tuple(output_shape), dtype=torch.bfloat16, device='cuda')
        self.context.set_tensor_address(output_name, output_tensor.data_ptr())

        # Run
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return output_tensor


# ============================================================================
# Global State Model Holder
# ============================================================================

class TRTTTSModelHolder:
    """Full Synthesis Pipeline using TRT"""

    def __init__(self):
        self.encoder_session = None
        self.trt_decoder = None
        self.predict_layer = None
        self.audio_embedding = None
        self.tokenizer = None
        self.audio_tokenizer = None
        self.args = None
        self.is_loaded = False

    def load_model(
        self,
        onnx_dir: str = "onnx_models_fp16_fixed",
        engine_path: str = "tensorrt_llm_implementation/engine_output/t5gemma_decoder_new.engine",
        weights_path: str = "weights/decoder_pmrope.bin",
    ):
        logger.info("Initializing TRT-TTS Pipeline...")
        
        # 1. Load Args
        args_path = os.path.join(os.path.dirname(weights_path), "model_args.json")
        with open(args_path, "r") as f:
            args_dict = json.load(f)
        self.args = type("Args", (), args_dict)()
        
        # 2. Tokenizer
        tokenizer_path = os.environ.get("TOKENIZER_PATH", getattr(self.args, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2"))
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 3. ONNX Encoder
        encoder_path = os.path.join(onnx_dir, "encoder.onnx")
        self.encoder_session = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])

        # 4. TRT Decoder
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT Engine not found: {engine_path}")
        self.trt_decoder = TRTDecoderWrapper(engine_path)

        # 5. Predict Layer & Audio Embedding (PyTorch side)
        # We need these to convert tokens -> embeds and hidden -> logits
        from models.t5gemma import T5GemmaVoiceModel
        temp_model = T5GemmaVoiceModel(self.args).to(dtype=torch.bfloat16, device='cuda')
        state_dict = torch.load(weights_path, map_location="cpu")
        temp_model.load_state_dict(state_dict, strict=False)
        
        self.audio_embedding = temp_model.audio_embedding[0].eval()
        self.predict_layer = temp_model.predict_layer[0].eval()
        self.audio_dropout = temp_model.audio_dropout.eval()
        self.progress_scale = temp_model.progress_scale
        
        del temp_model # Free memory
        torch.cuda.empty_cache()

        # 6. Audio Tokenizer
        from data.tokenizer import AudioTokenizer
        self.audio_tokenizer = AudioTokenizer(
            backend="xcodec2",
            model_name=getattr(self.args, "xcodec2_model_name", "NandemoGHS/Anime-XCodec2-44.1kHz-v2"),
            device="cuda",
        )

        self.is_loaded = True
        logger.info("TRT-TTS Pipeline loaded successfully!")

    def _build_position_ids(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        pos = torch.arange(max_len, device='cuda', dtype=torch.float32)[None, :]
        denom = (lengths.clamp(min=2).float() - 1.0)[:, None]
        return (pos / denom * self.progress_scale).masked_fill(pos >= lengths[:, None], 0.0)

    @torch.inference_mode()
    def synthesize(self, request: SynthesizeRequest):
        from inference_tts_utils import normalize_text_with_lang
        from duration_estimator import estimate_duration

        # 1. Normalize & Detect Lang
        normalized_text, lang_code = normalize_text_with_lang(request.text, request.language)
        
        # 2. Duration
        target_duration = request.target_duration
        if target_duration is None:
            target_duration = estimate_duration(target_text=normalized_text, target_lang=lang_code)
        
        target_total_tokens = int(target_duration * self.args.encodec_sr)
        max_tokens = min(target_total_tokens + int(self.args.encodec_sr), 2048)

        # 3. Encoder
        text_tokens = self.tokenizer.encode(normalized_text.strip(), add_special_tokens=False)
        # Add EOS/BOS if configured
        if getattr(self.args, "add_eos_to_text", 0): text_tokens.append(self.args.add_eos_to_text)
        if getattr(self.args, "add_bos_to_text", 0): text_tokens = [self.args.add_bos_to_text] + text_tokens
        
        input_ids_enc = np.array([text_tokens], dtype=np.int64)
        enc_out = self.encoder_session.run(None, {"input_ids": input_ids_enc, "attention_mask": np.ones_like(input_ids_enc)})[0]
        memory = torch.from_numpy(enc_out).to(device='cuda', dtype=torch.bfloat16)
        
        enc_len = torch.tensor([input_ids_enc.shape[1]], device='cuda')
        enc_pos_ids = self._build_position_ids(enc_len, input_ids_enc.shape[1])
        enc_mask = torch.ones((1, input_ids_enc.shape[1]), dtype=torch.int32, device='cuda')

        # 4. Generation Loop
        generated_tokens = []
        # BOS token
        current_tokens = torch.tensor([[self.args.empty_token]], device='cuda', dtype=torch.long)
        est_total = target_total_tokens + 1
        
        # Note: Current TRT engine doesn't have KV cache implemented in modeling.py
        # So we MUST pass the full sequence so far at each step.
        # This is O(N^2) but works with the existing engine.
        
        logger.info(f"Generating up to {max_tokens} tokens...")
        start_time = time.time()
        
        for i in range(max_tokens):
            # Prep inputs for this step
            cur_len = current_tokens.shape[1]
            pos_base = torch.arange(cur_len, device='cuda', dtype=torch.float32).unsqueeze(0)
            dec_pos_ids = pos_base / max(1, est_total - 1) * self.progress_scale
            
            trt_inputs = {
                "input_ids": current_tokens.to(torch.int32),
                "encoder_hidden_states": memory,
                "position_ids": dec_pos_ids,
                "encoder_position_ids": enc_pos_ids,
                "encoder_attention_mask": enc_mask
            }
            
            # Run TRT Decoder
            hidden_states = self.trt_decoder.run(trt_inputs)
            last_hidden = hidden_states[:, -1:, :] # [1, 1, hidden]
            
            # Predict & Sample
            logits = self.predict_layer(last_hidden).squeeze(0).squeeze(0)
            
            # Sampling logic
            if i == 0: logits[self.args.eog] = -1e9 # Prevent early EOG
            
            token = topk_sampling(logits, top_k=request.top_k, top_p=request.top_p, temperature=request.temperature)
            token_id = int(token.item())
            
            if token_id == self.args.eog or token_id == getattr(self.args, "eos", -1):
                break
                
            generated_tokens.append(token_id)
            current_tokens = torch.cat([current_tokens, token.unsqueeze(0)], dim=1)
            
            if (i+1) % 50 == 0: logger.info(f"Generated {i+1} tokens...")

        inf_time = time.time() - start_time
        logger.info(f"Generation complete: {len(generated_tokens)} tokens in {inf_time:.2f}s")

        # 5. Decode Audio
        if not generated_tokens:
            raise RuntimeError("No tokens generated")
            
        token_tensor = torch.tensor([[generated_tokens]], device='cuda', dtype=torch.long)
        audio = self.audio_tokenizer.decode(token_tensor)[0].cpu().numpy()
        
        return self.audio_tokenizer.sample_rate, audio.flatten(), inf_time

    def unload(self):
        self.is_loaded = False
        self.trt_decoder = None
        self.encoder_session = None
        torch.cuda.empty_cache()


# ============================================================================
# FastAPI App
# ============================================================================

model_holder = TRTTTSModelHolder()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use environment variables or defaults
    model_holder.load_model(
        onnx_dir=os.environ.get("ONNX_DIR", "onnx_models_fp16_fixed"),
        engine_path=os.environ.get("TRT_ENGINE", "tensorrt_llm_implementation/engine_output/t5gemma_decoder_new.engine"),
        weights_path=os.environ.get("WEIGHTS_PATH", "weights/decoder_pmrope.bin")
    )
    yield


app = FastAPI(title="T5Gemma-TTS TensorRT API", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "healthy", "trt_loaded": model_holder.is_loaded}

@app.post("/synthesize")
async def synthesize(request: SynthesizeRequest):
    if not model_holder.is_loaded: raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        sr, audio, inf_time = model_holder.synthesize(request)
        
        # Convert to WAV
        buffer = io.BytesIO()
        audio_int16 = (audio / (np.abs(audio).max() + 1e-7) * 32767).astype(np.int16)
        wav.write(buffer, sr, audio_int16)
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize_base64", response_model=SynthesizeResponse)
async def synthesize_base64(request: SynthesizeRequest):
    if not model_holder.is_loaded: raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        sr, audio, inf_time = model_holder.synthesize(request)
        
        buffer = io.BytesIO()
        audio_int16 = (audio / (np.abs(audio).max() + 1e-7) * 32767).astype(np.int16)
        wav.write(buffer, sr, audio_int16)
        buffer.seek(0)
        
        base64_audio = base64.b64encode(buffer.read()).decode("utf-8")
        return SynthesizeResponse(audio_base64=base64_audio, sample_rate=sr, inference_time=inf_time)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
