"""
T5Gemma-TTS FastAPI Server
Uses TensorRT engines directly for inference
"""

import os
import json
import torch
import tensorrt as trt
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from transformers import AutoTokenizer
import io
import scipy.io.wavfile as wavfile

app = FastAPI(title="T5Gemma-TTS API")

# Global model instances
encoder_ctx = None
decoder_ctx = None
vocoder_ctx = None
tokenizer = None
model_args = None


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = "ja"
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95


def load_engine(engine_path):
    """Load a TensorRT engine"""
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


@app.on_event("startup")
async def startup():
    global encoder_ctx, decoder_ctx, vocoder_ctx, tokenizer, model_args

    # Use /app as base path (mounted in Docker)
    base_path = "/app"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        f"{base_path}/tokenizer_local", local_files_only=True
    )

    print("Loading model args...")
    with open(f"{base_path}/weights/model_args.json") as f:
        model_args = json.load(f)

    print("Loading encoder engine...")
    _, encoder_ctx = load_engine(f"{base_path}/trt_weights/encoder_trt.engine")

    print("Loading decoder engine...")
    _, decoder_ctx = load_engine(
        f"{base_path}/trt_weights/t5gemma_decoder_with_lm_head.engine"
    )

    print("Loading vocoder engine...")
    _, vocoder_ctx = load_engine(f"{base_path}/trt_weights/vocoder.engine")

    print("All models loaded!")


@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    global encoder_ctx, decoder_ctx, vocoder_ctx, tokenizer, model_args

    try:
        # Tokenize
        inputs = tokenizer(request.text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(torch.int64).cuda()
        attention_mask = inputs["attention_mask"].to(torch.int64).cuda()

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_size = 2304

        # Run encoder
        encoder_ctx.set_input_shape("input_ids", input_ids.shape)
        encoder_ctx.set_input_shape("attention_mask", attention_mask.shape)

        enc_hidden = torch.empty(
            batch_size, seq_len, hidden_size, dtype=torch.float32, device="cuda"
        )

        encoder_ctx.set_tensor_address("input_ids", input_ids.data_ptr())
        encoder_ctx.set_tensor_address("attention_mask", attention_mask.data_ptr())
        encoder_ctx.set_tensor_address("encoder_hidden_states", enc_hidden.data_ptr())

        encoder_ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()

        # Decode (autoregressive)
        empty_token = model_args.get("empty_token", 65536)
        eog = model_args.get("eog", 65537)
        eos = model_args.get("eos", 65539)
        progress_scale = model_args.get("progress_scale", 2000.0)
        encodec_sr = model_args.get("encodec_sr", 50.0)

        # Estimate duration
        spp = {"ja": 0.12, "en": 0.08, "zh": 0.15}.get(request.language, 0.1)
        target_dur = len(request.text) * spp
        target_dur = max(target_dur, 1.0)
        max_tokens = min(int(target_dur * encodec_sr + 50), 2048)

        # Encoder positions
        enc_len = seq_len
        enc_pos_ids = (
            torch.arange(enc_len, dtype=torch.float32, device="cuda") / enc_len
        ) * progress_scale
        enc_pos_ids = enc_pos_ids.unsqueeze(0)

        # Convert encoder hidden to bfloat16
        enc_hidden_bf16 = enc_hidden.to(torch.bfloat16)
        enc_mask = attention_mask.to(torch.int32)

        # Start with empty token
        dec_input_ids = torch.tensor([[empty_token]], dtype=torch.int32, device="cuda")
        generated = []

        for step in range(max_tokens):
            cur_len = dec_input_ids.shape[1]
            est_total = max(1.0, target_dur * encodec_sr)

            dec_pos = (
                torch.arange(cur_len, dtype=torch.float32, device="cuda") / est_total
            ) * progress_scale
            dec_pos = dec_pos.unsqueeze(0)

            # Set shapes
            decoder_ctx.set_input_shape("input_ids", dec_input_ids.shape)
            decoder_ctx.set_input_shape("encoder_hidden_states", enc_hidden_bf16.shape)
            decoder_ctx.set_input_shape("position_ids", dec_pos.shape)
            decoder_ctx.set_input_shape("encoder_position_ids", enc_pos_ids.shape)
            decoder_ctx.set_input_shape("encoder_attention_mask", enc_mask.shape)

            logits = torch.empty(
                batch_size, cur_len, 65541, dtype=torch.bfloat16, device="cuda"
            )

            decoder_ctx.set_tensor_address("input_ids", dec_input_ids.data_ptr())
            decoder_ctx.set_tensor_address(
                "encoder_hidden_states", enc_hidden_bf16.data_ptr()
            )
            decoder_ctx.set_tensor_address("position_ids", dec_pos.data_ptr())
            decoder_ctx.set_tensor_address(
                "encoder_position_ids", enc_pos_ids.data_ptr()
            )
            decoder_ctx.set_tensor_address(
                "encoder_attention_mask", enc_mask.data_ptr()
            )
            decoder_ctx.set_tensor_address("logits", logits.data_ptr())

            decoder_ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()

            # Sample
            next_logits = logits[:, -1, :].float() / request.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(-1).item()

            if next_token in [eog, eos]:
                break

            generated.append(next_token)
            next_token_t = torch.tensor(
                [[next_token]], dtype=torch.int32, device="cuda"
            )
            dec_input_ids = torch.cat([dec_input_ids, next_token_t], dim=1)

        if not generated:
            raise HTTPException(status_code=500, detail="No audio tokens generated")

        # Run vocoder
        audio_tokens = torch.tensor(
            [generated], dtype=torch.int64, device="cuda"
        ).unsqueeze(1)

        vocoder_ctx.set_input_shape("codes", audio_tokens.shape)

        hop_size = 882
        audio_len = len(generated) * hop_size
        audio = torch.empty(1, 1, audio_len, dtype=torch.float32, device="cuda")

        vocoder_ctx.set_tensor_address("codes", audio_tokens.data_ptr())
        vocoder_ctx.set_tensor_address("audio", audio.data_ptr())

        vocoder_ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()

        # Convert to WAV
        audio_np = audio.cpu().numpy().flatten()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, 44100, audio_int16)
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
