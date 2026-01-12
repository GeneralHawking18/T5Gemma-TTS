import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM

model_name = "Aratako/T5Gemma-TTS-2b-2b"
print(f"Loading {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cpu")
decoder = model.decoder_module
layer = decoder.layers[0]

# Load inputs from golden_outputs
input_ids = torch.from_numpy(np.load("golden_outputs/input_ids.npy")).long()
encoder_hidden = torch.from_numpy(np.load("golden_outputs/encoder_hidden_states.npy")).bfloat16()

with torch.no_grad():
    # 1. Embedding
    # Note: decoder.embed_tokens is Identity due to pruning. Use audio_embedding.
    x = model.audio_embedding[0](input_ids)
    print(f"Embed shape: {x.shape}")
    print(f"Embed range: [{x.min().item():.4f}, {x.max().item():.4f}]")

    # 2. Norm 1
    print(f"Norm weight shape: {layer.pre_self_attn_layernorm.weight.shape}")
    normed = layer.pre_self_attn_layernorm(x)
    print(f"Norm1 range: [{normed.min().item():.4f}, {normed.max().item():.4f}]")

    # 3. Attention Projections
    q = layer.self_attn.q_proj(normed)
    k = layer.self_attn.k_proj(normed)
    v = layer.self_attn.v_proj(normed)
    print(f"Q (raw) range: [{q.min().item():.4f}, {q.max().item():.4f}]")
    print(f"K (raw) range: [{k.min().item():.4f}, {k.max().item():.4f}]")
    print(f"V (raw) range: [{v.min().item():.4f}, {v.max().item():.4f}]")

    # 4. Self Attention Output
    sa_out, _ = layer.self_attn(normed)
    print(f"SA Out range: [{sa_out.min().item():.4f}, {sa_out.max().item():.4f}]")
    
    # 5. Post SA Norm
    post_sa = layer.post_self_attn_layernorm(sa_out)
    print(f"Post SA Norm range: [{post_sa.min().item():.4f}, {post_sa.max().item():.4f}]")
