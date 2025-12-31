"""Compare Full Pipeline: Hybrid ONNX+PyTorch vs 4bit PyTorch.

This script traces intermediate outputs through both pipelines to identify
where the differences originate.
"""
import os
# Load HF_TOKEN
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

print("=" * 70)
print("FULL PIPELINE COMPARISON: Hybrid vs 4bit")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================================
# Load 4bit Model
# ========================================
print("\n[1] Loading 4bit PyTorch model...")
model_4bit = AutoModelForSeq2SeqLM.from_pretrained(
    "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model_4bit.eval()
cfg = model_4bit.config

tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
print(f"[2] Loading tokenizer from: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

# ========================================
# Load ONNX Encoder (CPU only to avoid GPU memory conflict)
# ========================================
print("\n[3] Loading ONNX Encoder...")
onnx_session = ort.InferenceSession(
    'onnx_models_fp16/encoder.onnx',
    providers=['CPUExecutionProvider']
)

# ========================================
# Test Input
# ========================================
text = "こんにちは、今日はいい天気ですね"
print(f"\nTest text: {text}")

# Tokenize (match 4bit logic: add_special_tokens=False + manual EOS)
add_eos = getattr(cfg, "add_eos_to_text", 1)
tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
if add_eos:
    tokens.append(add_eos)
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")

input_ids = torch.tensor([tokens], dtype=torch.long)
attention_mask = torch.ones_like(input_ids)

# ========================================
# Encoder Comparison
# ========================================
print("\n" + "=" * 70)
print("ENCODER COMPARISON")
print("=" * 70)

# ONNX Encoder
onnx_hidden = onnx_session.run(None, {
    "input_ids": input_ids.numpy().astype(np.int64),
    "attention_mask": attention_mask.numpy().astype(np.int64),
})[0]
print(f"\nONNX Encoder output shape: {onnx_hidden.shape}")
print(f"ONNX stats: mean={onnx_hidden.mean():.6f}, std={onnx_hidden.std():.6f}")

# PyTorch Encoder (4bit)
with torch.no_grad():
    input_ids_gpu = input_ids.to(model_4bit.device)
    attention_mask_gpu = attention_mask.to(model_4bit.device)

    # Compute PM-RoPE position IDs
    use_pm_rope = getattr(cfg, "use_pm_rope", True)
    progress_scale = getattr(cfg, "progress_scale", 2000.0)

    position_ids = None
    if use_pm_rope:
        x_lens = attention_mask_gpu.sum(dim=1)
        max_len = input_ids_gpu.shape[1]
        pos = torch.arange(max_len, device=model_4bit.device, dtype=torch.float32)[None, :]
        denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
        position_ids = pos / denom * progress_scale
        mask = pos < x_lens[:, None]
        position_ids = position_ids.masked_fill(~mask, 0.0)

    # Get encoder
    if hasattr(model_4bit, 'backbone'):
        encoder = model_4bit.backbone.model.encoder
    elif hasattr(model_4bit.model, 'encoder'):
        encoder = model_4bit.model.encoder
    else:
        encoder = model_4bit.model.model.encoder

    pt_encoder_out = encoder(
        input_ids=input_ids_gpu,
        attention_mask=attention_mask_gpu,
        position_ids=position_ids,
    )
    pt_hidden = pt_encoder_out.last_hidden_state.float().cpu().numpy()

print(f"PyTorch Encoder output shape: {pt_hidden.shape}")
print(f"PyTorch stats: mean={pt_hidden.mean():.6f}, std={pt_hidden.std():.6f}")

# Compare
diff = np.abs(onnx_hidden - pt_hidden)
cos_sim = np.dot(onnx_hidden.flatten(), pt_hidden.flatten()) / (
    np.linalg.norm(onnx_hidden.flatten()) * np.linalg.norm(pt_hidden.flatten())
)
print(f"\nEncoder Comparison:")
print(f"  Max diff: {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")
print(f"  Cosine similarity: {cos_sim:.6f}")

if cos_sim > 0.99:
    print("  ✅ Encoder outputs MATCH well")
else:
    print("  ❌ Encoder outputs DIFFER significantly!")

# ========================================
# Decoder First Step Comparison
# ========================================
print("\n" + "=" * 70)
print("DECODER FIRST STEP COMPARISON")
print("=" * 70)

# Setup target duration
target_duration = 5.0  # seconds
encodec_sr = getattr(cfg, "encodec_sr", 50.0)
target_total = int(target_duration * encodec_sr)
print(f"Target duration: {target_duration}s -> {target_total} audio frames")

# Common parameters
empty_token = getattr(cfg, "empty_token", 65536)
print(f"Empty/BOS token: {empty_token}")

# ========================================
# 4bit Decoder First Step
# ========================================
print("\n--- 4bit PyTorch Decoder ---")
with torch.no_grad():
    # Get decoder components
    if hasattr(model_4bit, 'backbone'):
        decoder = model_4bit.backbone.model.decoder
        audio_embedding = model_4bit.backbone.audio_embedding[0]
        audio_dropout = model_4bit.backbone.audio_dropout
        predict_layer = model_4bit.backbone.predict_layer[0]
    else:
        decoder = model_4bit.model.model.decoder
        audio_embedding = model_4bit.model.audio_embedding[0]
        audio_dropout = model_4bit.model.audio_dropout
        predict_layer = model_4bit.model.predict_layer[0]

    # Prepare encoder hidden states (use 4bit encoder output)
    encoder_hidden = pt_encoder_out.last_hidden_state

    # Create BOS token
    bos = torch.full((1, 1), empty_token, dtype=torch.long, device=model_4bit.device)

    # Embed and dropout
    embedded_y = audio_embedding(bos)
    embedded_y = audio_dropout(embedded_y)  # Important!

    # PM-RoPE for decoder
    est_total = target_total + 1  # +1 for BOS
    x_lens = attention_mask_gpu.sum(dim=1)
    max_len = input_ids_gpu.shape[1]

    # Encoder position IDs
    enc_pos = torch.arange(max_len, device=model_4bit.device, dtype=torch.float32)[None, :]
    denom_enc = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
    pm_encoder_position_ids = enc_pos / denom_enc * progress_scale
    enc_mask = enc_pos < x_lens[:, None]
    pm_encoder_position_ids = pm_encoder_position_ids.masked_fill(~enc_mask, 0.0)

    # Decoder position IDs
    cur_len = 1
    dec_pos = torch.arange(cur_len, device=model_4bit.device, dtype=torch.float32)[None, :]
    pm_decoder_position_ids = dec_pos / (est_total - 1) * progress_scale

    print(f"  Decoder position IDs: {pm_decoder_position_ids[0].tolist()}")
    print(f"  est_total: {est_total}")

    # Run decoder
    decoder_attention_mask = torch.ones((1, cur_len), dtype=torch.long, device=model_4bit.device)

    decoder_output_4bit = decoder(
        inputs_embeds=embedded_y,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden,
        encoder_attention_mask=attention_mask_gpu,
        position_ids=pm_decoder_position_ids,
        pm_decoder_position_ids=pm_decoder_position_ids,
        pm_encoder_position_ids=pm_encoder_position_ids,
        use_cache=True,
    )

    hidden_4bit = decoder_output_4bit.last_hidden_state[:, -1:, :]
    logits_4bit = predict_layer(hidden_4bit)

    print(f"  Hidden state shape: {hidden_4bit.shape}")
    print(f"  Hidden stats: mean={hidden_4bit.float().mean().item():.6f}, std={hidden_4bit.float().std().item():.6f}")
    print(f"  Logits shape: {logits_4bit.shape}")
    print(f"  Top 5 logits: {logits_4bit[0, 0, :5].float().tolist()}")

# ========================================
# Hybrid Decoder First Step (using ONNX encoder output)
# ========================================
print("\n--- Hybrid Decoder (ONNX encoder + PyTorch decoder) ---")

# Load hybrid components from inference_hybrid_complete
import json
with open("onnx_models_fp16/model_args.json") as f:
    args = json.load(f)

# Create args object
class Args:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)
hybrid_args = Args(args)

# Load decoder weights
from inference_hybrid_complete import HybridT5GemmaTTS

# Initialize hybrid TTS (this loads the decoder)
print("\n  Loading hybrid decoder...")
hybrid_tts = HybridT5GemmaTTS(
    onnx_dir="onnx_models_fp16",
    decoder_weights="weights/decoder_pmrope.bin",
    device=device,
)

with torch.no_grad():
    # Convert ONNX encoder output to tensor
    encoder_hidden_hybrid = torch.from_numpy(onnx_hidden).to(device=device, dtype=hybrid_tts.dtype)
    attention_mask_hybrid = attention_mask.to(device=device)

    # Create BOS token
    bos_hybrid = torch.full((1, 1), empty_token, dtype=torch.long, device=device)

    # Embed and dropout
    embedded_y_hybrid = hybrid_tts.model.audio_embedding[0](bos_hybrid)
    embedded_y_hybrid = hybrid_tts.model.audio_dropout(embedded_y_hybrid)

    # PM-RoPE for decoder (same calculation)
    x_lens_hybrid = attention_mask_hybrid.sum(dim=1)
    max_len_hybrid = encoder_hidden_hybrid.shape[1]

    enc_pos_hybrid = torch.arange(max_len_hybrid, device=device, dtype=torch.float32)[None, :]
    denom_enc_hybrid = (x_lens_hybrid.clamp(min=2).to(torch.float32) - 1.0)[:, None]
    pm_encoder_position_ids_hybrid = enc_pos_hybrid / denom_enc_hybrid * progress_scale
    enc_mask_hybrid = enc_pos_hybrid < x_lens_hybrid[:, None]
    pm_encoder_position_ids_hybrid = pm_encoder_position_ids_hybrid.masked_fill(~enc_mask_hybrid, 0.0)

    # Decoder position IDs
    dec_pos_hybrid = torch.arange(cur_len, device=device, dtype=torch.float32)[None, :]
    pm_decoder_position_ids_hybrid = dec_pos_hybrid / (est_total - 1) * progress_scale

    print(f"  Decoder position IDs: {pm_decoder_position_ids_hybrid[0].tolist()}")
    print(f"  est_total: {est_total}")

    # Run decoder
    decoder_attention_mask_hybrid = torch.ones((1, cur_len), dtype=torch.long, device=device)

    decoder_output_hybrid = hybrid_tts.model.decoder(
        inputs_embeds=embedded_y_hybrid,
        attention_mask=decoder_attention_mask_hybrid,
        encoder_hidden_states=encoder_hidden_hybrid,
        encoder_attention_mask=attention_mask_hybrid,
        position_ids=pm_decoder_position_ids_hybrid,
        pm_decoder_position_ids=pm_decoder_position_ids_hybrid,
        pm_encoder_position_ids=pm_encoder_position_ids_hybrid,
        use_cache=True,
    )

    hidden_hybrid = decoder_output_hybrid.last_hidden_state[:, -1:, :]
    logits_hybrid = hybrid_tts.model.predict_layer[0](hidden_hybrid)

    print(f"  Hidden state shape: {hidden_hybrid.shape}")
    print(f"  Hidden stats: mean={hidden_hybrid.float().mean().item():.6f}, std={hidden_hybrid.float().std().item():.6f}")
    print(f"  Logits shape: {logits_hybrid.shape}")
    print(f"  Top 5 logits: {logits_hybrid[0, 0, :5].float().tolist()}")

# ========================================
# Compare Decoder Outputs
# ========================================
print("\n" + "=" * 70)
print("DECODER OUTPUT COMPARISON")
print("=" * 70)

hidden_4bit_np = hidden_4bit.float().cpu().numpy()
hidden_hybrid_np = hidden_hybrid.float().cpu().numpy()

diff_hidden = np.abs(hidden_4bit_np - hidden_hybrid_np)
cos_sim_hidden = np.dot(hidden_4bit_np.flatten(), hidden_hybrid_np.flatten()) / (
    np.linalg.norm(hidden_4bit_np.flatten()) * np.linalg.norm(hidden_hybrid_np.flatten())
)

print(f"Hidden State Comparison:")
print(f"  Max diff: {diff_hidden.max():.6f}")
print(f"  Mean diff: {diff_hidden.mean():.6f}")
print(f"  Cosine similarity: {cos_sim_hidden:.6f}")

logits_4bit_np = logits_4bit.float().cpu().numpy()
logits_hybrid_np = logits_hybrid.float().cpu().numpy()

diff_logits = np.abs(logits_4bit_np - logits_hybrid_np)
cos_sim_logits = np.dot(logits_4bit_np.flatten(), logits_hybrid_np.flatten()) / (
    np.linalg.norm(logits_4bit_np.flatten()) * np.linalg.norm(logits_hybrid_np.flatten())
)

print(f"\nLogits Comparison:")
print(f"  Max diff: {diff_logits.max():.6f}")
print(f"  Mean diff: {diff_logits.mean():.6f}")
print(f"  Cosine similarity: {cos_sim_logits:.6f}")

# Check argmax (predicted token)
pred_4bit = logits_4bit.argmax(dim=-1).item()
pred_hybrid = logits_hybrid.argmax(dim=-1).item()
print(f"\nPredicted next token:")
print(f"  4bit:   {pred_4bit}")
print(f"  Hybrid: {pred_hybrid}")
print(f"  Match: {'✅' if pred_4bit == pred_hybrid else '❌'}")

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if cos_sim > 0.99 and cos_sim_hidden > 0.99 and pred_4bit == pred_hybrid:
    print("✅ Pipeline outputs are consistent!")
    print("   Any audio differences may be due to sampling randomness.")
elif cos_sim > 0.99 and cos_sim_hidden < 0.99:
    print("❌ ISSUE: Encoder outputs match but decoder outputs differ!")
    print("   Check: decoder weights, precision, or position embeddings.")
elif cos_sim < 0.99:
    print("❌ ISSUE: Encoder outputs differ significantly!")
    print("   Check: ONNX export or position ID computation.")
else:
    print("⚠️ Outputs differ - further investigation needed.")
    print(f"   Encoder cosine sim: {cos_sim:.6f}")
    print(f"   Decoder hidden cosine sim: {cos_sim_hidden:.6f}")
    print(f"   Token prediction match: {pred_4bit == pred_hybrid}")
