"""Compare ONNX encoder (from non-quantized) vs non-quantized PyTorch encoder.

The ONNX was exported from 'Aratako/T5Gemma-TTS-2b-2b' (not 4bit).
We should compare against the same non-quantized model.
"""
import os
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading NON-QUANTIZED model (same as ONNX source)...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b",  # Non-quantized version!
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,  # Same as ONNX export dtype
)
model.eval()
cfg = model.config

tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

print("\nLoading ONNX encoder...")
onnx_session = ort.InferenceSession(
    'onnx_models_fp16/encoder.onnx',
    providers=['CPUExecutionProvider']
)

# Test texts
test_texts = [
    "こんにちは",
    "こんにちは、今日はいい天気ですね",
    "Hello world, this is a test",
]

add_eos = getattr(cfg, "add_eos_to_text", 1)
progress_scale = getattr(cfg, "progress_scale", 2000.0)

for text in test_texts:
    print(f"\n{'='*60}")
    print(f"Text: {text}")

    # Tokenize (same as ONNX encoder wrapper)
    tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
    if add_eos:
        tokens.append(add_eos)
    print(f"Tokens: {len(tokens)} total")

    input_ids = torch.tensor([tokens], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # ONNX encoder
    onnx_out = onnx_session.run(None, {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    })[0]

    # PyTorch encoder (non-quantized)
    with torch.no_grad():
        input_ids_gpu = input_ids.to(model.device)
        attention_mask_gpu = attention_mask.to(model.device)

        # Compute PM-RoPE position IDs
        x_lens = attention_mask_gpu.sum(dim=1)
        max_len = input_ids_gpu.shape[1]
        pos = torch.arange(max_len, device=model.device, dtype=torch.float32)[None, :]
        denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
        position_ids = pos / denom * progress_scale
        mask = pos < x_lens[:, None]
        position_ids = position_ids.masked_fill(~mask, 0.0)

        # Get encoder - check model structure
        if hasattr(model, 'backbone'):
            encoder = model.backbone.model.encoder
        elif hasattr(model, 'model'):
            if hasattr(model.model, 'encoder'):
                encoder = model.model.encoder
            else:
                encoder = model.model.model.encoder
        else:
            raise AttributeError("Cannot find encoder")

        pt_out = encoder(
            input_ids=input_ids_gpu,
            attention_mask=attention_mask_gpu,
            position_ids=position_ids,
        ).last_hidden_state.float().cpu().numpy()

    # Compare
    cos_sim = np.dot(onnx_out.flatten(), pt_out.flatten()) / (
        np.linalg.norm(onnx_out.flatten()) * np.linalg.norm(pt_out.flatten()) + 1e-8
    )
    diff = np.abs(onnx_out - pt_out)

    print(f"ONNX:    mean={onnx_out.mean():.4f}, std={onnx_out.std():.4f}")
    print(f"PyTorch: mean={pt_out.mean():.4f}, std={pt_out.std():.4f}")
    print(f"Max diff: {diff.max():.4f}, Mean diff: {diff.mean():.4f}")
    print(f"Cosine similarity: {cos_sim:.6f}")

    if cos_sim > 0.999:
        print("✅ EXCELLENT MATCH")
    elif cos_sim > 0.99:
        print("✅ Good match")
    elif cos_sim > 0.95:
        print("⚠️ Acceptable")
    else:
        print("❌ SIGNIFICANT DIFFERENCE")
