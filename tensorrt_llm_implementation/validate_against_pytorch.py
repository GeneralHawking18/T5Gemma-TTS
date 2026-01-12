"""
Validate TensorRT-LLM implementation against original PyTorch model.
Compares outputs block-by-block to identify discrepancies.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

def load_original_model():
    """Load original T5Gemma model from HuggingFace."""
    print("Loading original model from aratako/t5gemma-2b-2b...")
    model = AutoModelForCausalLM.from_pretrained(
        "aratako/t5gemma-2b-2b",
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).cuda()
    model.eval()
    return model

def load_trt_weights():
    """Load TensorRT weights for comparison."""
    print("Loading TRT weights from weights.npz...")
    weights = np.load('/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/trt_weights/weights.npz')
    return weights

def compare_weights(pytorch_model, trt_weights):
    """Compare weight shapes and values between PyTorch and TRT."""
    print("\n" + "="*60)
    print("WEIGHT COMPARISON")
    print("="*60)

    decoder = pytorch_model.model.decoder

    # Compare embedding
    print("\n--- Embedding ---")
    pt_embed = decoder.embed_tokens.weight.data.cpu().numpy()
    trt_embed = trt_weights['embed_tokens.weight']
    print(f"PyTorch embed shape: {pt_embed.shape}")
    print(f"TRT embed shape:     {trt_embed.shape}")
    if pt_embed.shape == trt_embed.shape:
        diff = np.abs(pt_embed.astype(np.float32) - trt_embed.astype(np.float32)).max()
        print(f"Max difference: {diff:.6e}")
    else:
        print("SHAPE MISMATCH!")

    # Compare layer 0 weights
    print("\n--- Layer 0 Self-Attention ---")
    layer0 = decoder.layers[0]

    comparisons = [
        ('sa_norm', layer0.pre_self_attn_layernorm.weight, 'layers.0.sa_norm.weight'),
        ('self_attn.q', layer0.self_attn.q.weight, 'layers.0.self_attn.q.weight'),
        ('self_attn.k', layer0.self_attn.k.weight, 'layers.0.self_attn.k.weight'),
        ('self_attn.v', layer0.self_attn.v.weight, 'layers.0.self_attn.v.weight'),
        ('self_attn.o', layer0.self_attn.o.weight, 'layers.0.self_attn.o.weight'),
    ]

    for name, pt_weight, trt_key in comparisons:
        pt_w = pt_weight.data.cpu().numpy()
        if trt_key in trt_weights:
            trt_w = trt_weights[trt_key]
            print(f"{name}: PT {pt_w.shape} vs TRT {trt_w.shape}", end=" ")
            if pt_w.shape == trt_w.shape:
                diff = np.abs(pt_w.astype(np.float32) - trt_w.astype(np.float32)).max()
                print(f"max_diff={diff:.6e}")
            else:
                print("SHAPE MISMATCH!")
        else:
            print(f"{name}: TRT key '{trt_key}' NOT FOUND!")

def test_embedding_only(pytorch_model, trt_weights):
    """Test just the embedding layer."""
    print("\n" + "="*60)
    print("EMBEDDING TEST")
    print("="*60)

    # Create test input
    input_ids = torch.randint(0, 1000, (1, 32), dtype=torch.long).cuda()

    # PyTorch embedding
    decoder = pytorch_model.model.decoder
    pt_embed_out = decoder.embed_tokens(input_ids)
    print(f"PyTorch embedding output: {pt_embed_out.shape}")
    print(f"  Range: [{pt_embed_out.min():.4f}, {pt_embed_out.max():.4f}]")
    print(f"  Sample: {pt_embed_out[0, 0, :5].tolist()}")

    # Manual embedding using TRT weights
    trt_embed_weight = torch.from_numpy(trt_weights['embed_tokens.weight']).cuda()
    trt_embed_out = F.embedding(input_ids, trt_embed_weight)
    print(f"\nTRT weights embedding output: {trt_embed_out.shape}")
    print(f"  Range: [{trt_embed_out.min():.4f}, {trt_embed_out.max():.4f}]")
    print(f"  Sample: {trt_embed_out[0, 0, :5].tolist()}")

    # Compare
    diff = (pt_embed_out - trt_embed_out).abs()
    print(f"\nDifference: max={diff.max():.6e}, mean={diff.mean():.6e}")

def test_layer_norm(pytorch_model, trt_weights):
    """Test LayerNorm operation."""
    print("\n" + "="*60)
    print("LAYER NORM TEST")
    print("="*60)

    decoder = pytorch_model.model.decoder
    layer0 = decoder.layers[0]

    # Test input
    x = torch.randn(1, 32, 2304, dtype=torch.float16).cuda()

    # PyTorch LayerNorm
    pt_ln = layer0.pre_self_attn_layernorm
    pt_out = pt_ln(x)
    print(f"PyTorch LN output: {pt_out.shape}")
    print(f"  Range: [{pt_out.min():.4f}, {pt_out.max():.4f}]")
    print(f"  Has NaN: {torch.isnan(pt_out).any()}")

    # Manual LayerNorm with TRT weights
    trt_weight = torch.from_numpy(trt_weights['layers.0.sa_norm.weight']).cuda()
    # RMSNorm (T5-style, no bias)
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + 1e-6)
    trt_out = trt_weight * x_norm
    print(f"\nManual RMSNorm output: {trt_out.shape}")
    print(f"  Range: [{trt_out.min():.4f}, {trt_out.max():.4f}]")
    print(f"  Has NaN: {torch.isnan(trt_out).any()}")

    diff = (pt_out - trt_out).abs()
    print(f"\nDifference: max={diff.max():.6e}, mean={diff.mean():.6e}")

def test_full_layer(pytorch_model, trt_weights, layer_idx=0):
    """Test a full decoder layer."""
    print("\n" + "="*60)
    print(f"FULL LAYER {layer_idx} TEST")
    print("="*60)

    decoder = pytorch_model.model.decoder
    layer = decoder.layers[layer_idx]

    # Test inputs
    batch_size = 1
    dec_seq = 32
    enc_seq = 64
    hidden_size = 2304

    hidden_states = torch.randn(batch_size, dec_seq, hidden_size, dtype=torch.float16).cuda()
    encoder_hidden_states = torch.randn(batch_size, enc_seq, hidden_size, dtype=torch.float16).cuda()

    # Position IDs (for PM-RoPE, these should be progress values 0-1)
    position_ids = torch.linspace(0, 1, dec_seq).unsqueeze(0).cuda()
    encoder_position_ids = torch.linspace(0, 1, enc_seq).unsqueeze(0).cuda()

    print(f"Input hidden_states: {hidden_states.shape}, range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
    print(f"Input encoder_hidden: {encoder_hidden_states.shape}")

    # Run through PyTorch layer
    with torch.no_grad():
        # Check intermediate steps

        # 1. Pre-self-attn layernorm
        normed = layer.pre_self_attn_layernorm(hidden_states)
        print(f"\n1. After pre_self_attn_layernorm:")
        print(f"   Range: [{normed.min():.4f}, {normed.max():.4f}]")
        print(f"   Has NaN: {torch.isnan(normed).any()}")

        # 2. Self attention Q/K/V projections
        q = layer.self_attn.q(normed)
        k = layer.self_attn.k(normed)
        v = layer.self_attn.v(normed)
        print(f"\n2. After Q/K/V projections:")
        print(f"   Q: range=[{q.min():.4f}, {q.max():.4f}], NaN={torch.isnan(q).any()}")
        print(f"   K: range=[{k.min():.4f}, {k.max():.4f}], NaN={torch.isnan(k).any()}")
        print(f"   V: range=[{v.min():.4f}, {v.max():.4f}], NaN={torch.isnan(v).any()}")

        # 3. Full self attention
        sa_out = layer.self_attn(
            normed,
            position_ids=position_ids,
            encoder_position_ids=encoder_position_ids
        )
        if isinstance(sa_out, tuple):
            sa_out = sa_out[0]
        print(f"\n3. After self_attn:")
        print(f"   Range: [{sa_out.min():.4f}, {sa_out.max():.4f}]")
        print(f"   Has NaN: {torch.isnan(sa_out).any()}")

        # 4. Residual
        hidden_states_after_sa = hidden_states + sa_out
        print(f"\n4. After SA residual:")
        print(f"   Range: [{hidden_states_after_sa.min():.4f}, {hidden_states_after_sa.max():.4f}]")
        print(f"   Has NaN: {torch.isnan(hidden_states_after_sa).any()}")

        # 5. Cross attention
        ca_normed = layer.pre_cross_attn_layernorm(hidden_states_after_sa)
        ca_out = layer.cross_attn(
            ca_normed,
            key_value_states=encoder_hidden_states,
            position_ids=position_ids,
            encoder_position_ids=encoder_position_ids
        )
        if isinstance(ca_out, tuple):
            ca_out = ca_out[0]
        print(f"\n5. After cross_attn:")
        print(f"   Range: [{ca_out.min():.4f}, {ca_out.max():.4f}]")
        print(f"   Has NaN: {torch.isnan(ca_out).any()}")

        # 6. MLP
        hidden_states_after_ca = hidden_states_after_sa + ca_out
        ff_normed = layer.pre_feedforward_layernorm(hidden_states_after_ca)
        ff_out = layer.mlp(ff_normed)
        print(f"\n6. After MLP:")
        print(f"   Range: [{ff_out.min():.4f}, {ff_out.max():.4f}]")
        print(f"   Has NaN: {torch.isnan(ff_out).any()}")

        # 7. Final output
        layer_out = hidden_states_after_ca + ff_out
        print(f"\n7. Final layer output:")
        print(f"   Range: [{layer_out.min():.4f}, {layer_out.max():.4f}]")
        print(f"   Has NaN: {torch.isnan(layer_out).any()}")

def test_rope(pytorch_model):
    """Test RoPE (Rotary Position Embedding) implementation."""
    print("\n" + "="*60)
    print("PM-ROPE TEST")
    print("="*60)

    decoder = pytorch_model.model.decoder
    layer0 = decoder.layers[0]
    self_attn = layer0.self_attn

    # Check if there's a rotary embedding
    if hasattr(self_attn, 'rotary_emb'):
        print("Found rotary_emb in self_attn")
        rotary = self_attn.rotary_emb
        print(f"  Type: {type(rotary)}")
        print(f"  Dim: {rotary.dim if hasattr(rotary, 'dim') else 'N/A'}")
    else:
        print("No rotary_emb found - checking for PM-RoPE")

    # Test manual RoPE computation
    head_dim = 64
    seq_len = 32
    batch_size = 1
    num_heads = 32

    # Create test tensor [B, S, H, D]
    x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float16).cuda()
    position_ids = torch.linspace(0, 1, seq_len).unsqueeze(0).cuda()

    print(f"\nTest input x: {x.shape}")
    print(f"Position IDs: {position_ids.shape}, range=[{position_ids.min():.4f}, {position_ids.max():.4f}]")

    # Compute inv_freq (same as in modeling.py)
    import math
    half_dim = head_dim // 2
    idx = torch.arange(0, half_dim, dtype=torch.float32).cuda() * 2.0
    neg_log_scale = -math.log(10000.0) / float(head_dim)
    inv_freq = torch.exp(idx * neg_log_scale)
    print(f"\ninv_freq: {inv_freq.shape}")
    print(f"  Range: [{inv_freq.min():.6f}, {inv_freq.max():.6f}]")
    print(f"  Sample: {inv_freq[:5].tolist()}")

    # Compute angles
    pos_expanded = position_ids.unsqueeze(-1)  # [B, S, 1]
    inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(0)  # [1, 1, D/2]
    angles = pos_expanded * inv_freq_expanded  # [B, S, D/2]
    print(f"\nAngles: {angles.shape}")
    print(f"  Range: [{angles.min():.6f}, {angles.max():.6f}]")

    # Full angles
    full_angles = torch.cat([angles, angles], dim=-1)  # [B, S, D]
    print(f"Full angles: {full_angles.shape}")

    # Cos/Sin
    cos_emb = torch.cos(full_angles).half()
    sin_emb = torch.sin(full_angles).half()
    print(f"\ncos_emb: range=[{cos_emb.min():.4f}, {cos_emb.max():.4f}], NaN={torch.isnan(cos_emb).any()}")
    print(f"sin_emb: range=[{sin_emb.min():.4f}, {sin_emb.max():.4f}], NaN={torch.isnan(sin_emb).any()}")

    # Apply RoPE
    cos_emb = cos_emb.unsqueeze(2)  # [B, S, 1, D]
    sin_emb = sin_emb.unsqueeze(2)  # [B, S, 1, D]

    # Rotate half
    x1 = x[..., :head_dim//2]
    x2 = x[..., head_dim//2:]
    rotated = torch.cat([-x2, x1], dim=-1)

    output = x * cos_emb + rotated * sin_emb
    print(f"\nRoPE output: {output.shape}")
    print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"  Has NaN: {torch.isnan(output).any()}")

def main():
    print("="*60)
    print("TensorRT-LLM vs PyTorch Validation")
    print("="*60)

    # Load models
    pytorch_model = load_original_model()
    trt_weights = load_trt_weights()

    print(f"\nPyTorch model type: {type(pytorch_model)}")
    print(f"TRT weights keys: {len(trt_weights.files)} tensors")

    # Run tests
    compare_weights(pytorch_model, trt_weights)
    test_embedding_only(pytorch_model, trt_weights)
    test_layer_norm(pytorch_model, trt_weights)
    test_rope(pytorch_model)
    test_full_layer(pytorch_model, trt_weights, layer_idx=0)

    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
