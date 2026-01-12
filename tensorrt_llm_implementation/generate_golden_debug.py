import torch
import numpy as np
import os
import sys
import json
import shutil

# Add parent directory to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Paths
BASE_DIR = PROJECT_ROOT
WEIGHTS_DIR = f"{BASE_DIR}/weights"
GOLDEN_DIR = f"{BASE_DIR}/golden_outputs"

def load_local_model():
    """Load local T5GemmaVoiceModel with decoder_pmrope.bin weights."""
    from models.t5gemma import T5GemmaVoiceModel

    # Load model args
    args_path = f"{WEIGHTS_DIR}/model_args.json"
    print(f"Loading args from {args_path}...")
    with open(args_path, 'r') as f:
        args_dict = json.load(f)
    args = type("Args", (), args_dict)()

    # Monkeypatch to avoid downloading backbone weights
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    original_from_pretrained = AutoModelForSeq2SeqLM.from_pretrained
    
    def no_download_from_pretrained(model_name, **kwargs):
        print(f"  Intercepted download for {model_name}. Initializing from config...")
        config = AutoConfig.from_pretrained(model_name)
        dtype = kwargs.get('torch_dtype', torch.float32)
        with torch.device("meta"):
            model = AutoModelForSeq2SeqLM.from_config(config)
        model.to_empty(device="cpu")
        model.to(dtype=dtype)
        return model
    
    import transformers
    transformers.AutoModelForSeq2SeqLM.from_pretrained = no_download_from_pretrained

    try:
        print("Creating T5GemmaVoiceModel...")
        model = T5GemmaVoiceModel(args)
    finally:
        transformers.AutoModelForSeq2SeqLM.from_pretrained = original_from_pretrained

    # Load weights
    decoder_weights_path = f"{WEIGHTS_DIR}/decoder_pmrope.bin"
    print(f"Loading decoder weights from {decoder_weights_path}...")
    state_dict = torch.load(decoder_weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # FIX: Always re-initialize inv_freq and clear cache because to_empty() leaves garbage
    if hasattr(model, "decoder_module") and hasattr(model.decoder_module, "rotary_emb"):
        rotary = model.decoder_module.rotary_emb
        print("Forcing re-initialization of rotary_emb...")
        
        # Print what was there (for debug)
        if hasattr(rotary, "inv_freq"):
            print(f"  Old inv_freq: range=[{rotary.inv_freq.min():.4e}, {rotary.inv_freq.max():.4e}]")
            
        # Deduce dim from inv_freq shape (dim/2)
        dim = rotary.inv_freq.shape[0] * 2
        print(f"  Duced dim={dim}")
        
        # Standard RoPE calc
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        rotary.inv_freq = inv_freq.to(device=rotary.inv_freq.device)
        
        # CRITICAL: Reset cache so it gets recomputed with valid inv_freq
        rotary.cos_cached = None
        rotary.sin_cached = None
        
        # Pre-compute for max length to ensure it's ready
        # T5GemmaRotaryEmbedding usually has _set_cos_sin_cache
        if hasattr(rotary, "_set_cos_sin_cache"):
            print("  Calling _set_cos_sin_cache...")
            rotary._set_cos_sin_cache(seq_len=2048, device=rotary.inv_freq.device, dtype=torch.float32)

    # Move to GPU with bfloat16
    model = model.to(dtype=torch.bfloat16, device='cuda')
    model.eval()
    
    # Debug: Print Config
    config = model.backbone.config
    print("\nModel Config:")
    print(f"  d_model: {getattr(config, 'd_model', 'N/A')}")
    print(f"  hidden_size: {getattr(config, 'hidden_size', 'N/A')}")
    print(f"  num_heads: {getattr(config, 'num_heads', getattr(config, 'num_attention_heads', 'N/A'))}")
    print(f"  d_kv: {getattr(config, 'd_kv', 'N/A')}")
    print(f"  head_dim: {getattr(config, 'head_dim', 'N/A')}")
    print(f"  num_kv_heads: {getattr(config, 'num_kv_heads', getattr(config, 'num_key_value_heads', 'N/A'))}")
    
    return model

def generate_golden():
    # Clean up
    for d in [GOLDEN_DIR, f"{BASE_DIR}/validation_outputs"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(GOLDEN_DIR)
    os.makedirs(f"{BASE_DIR}/validation_outputs", exist_ok=True)
    
    print("Loading PyTorch model...")
    model = load_local_model()
    
    # 1. Prepare Inputs (Deterministic)
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 1
    dec_seq_len = 32
    enc_seq_len = 64
    hidden_size = 2304
    
    # Use standard vocab range for audio tokens (0-65535)
    input_ids = torch.randint(0, 1000, (batch_size, dec_seq_len)).long().cuda()
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, hidden_size, dtype=torch.bfloat16).cuda()
    
    # Position IDs
    position_ids = torch.linspace(0, 1, dec_seq_len, dtype=torch.float32).unsqueeze(0).cuda()
    encoder_position_ids = torch.linspace(0, 1, enc_seq_len, dtype=torch.float32).unsqueeze(0).cuda()

    print("\nSaving inputs...")
    np.save(f"{GOLDEN_DIR}/input_ids.npy", input_ids.cpu().numpy())
    np.save(f"{GOLDEN_DIR}/encoder_hidden_states.npy", encoder_hidden_states.float().cpu().numpy())
    np.save(f"{GOLDEN_DIR}/position_ids.npy", position_ids.cpu().numpy())
    np.save(f"{GOLDEN_DIR}/encoder_position_ids.npy", encoder_position_ids.cpu().numpy())
    
    # Also copy to validation_outputs so other scripts can find them
    for f in ["input_ids.npy", "encoder_hidden_states.npy", "position_ids.npy", "encoder_position_ids.npy"]:
        shutil.copy(f"{GOLDEN_DIR}/{f}", f"{BASE_DIR}/validation_outputs/{f}")

    # 2. Run Forward Pass Layer-by-Layer
    print("\nRunning forward pass...")
    decoder = model.decoder_module
    
    with torch.no_grad():
        # A. Embedding
        audio_embedding = model.audio_embedding[0]
        hidden_states = audio_embedding(input_ids)
        
        print(f"Embedding output: {hidden_states.shape} range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
        if torch.isnan(hidden_states).any():
            print("CRITICAL: Embedding produced NaNs!")
        np.save(f"{GOLDEN_DIR}/embedding_out.npy", hidden_states.float().cpu().numpy())
        
        # B. Rotary Embeddings
        sa_position_ids = torch.arange(dec_seq_len, dtype=torch.long, device='cuda').unsqueeze(0)
        
        position_embeddings = None
        if hasattr(decoder, "rotary_emb"):
            position_embeddings = decoder.rotary_emb(hidden_states, sa_position_ids)
            # Check rotary
            if position_embeddings:
                cos, sin = position_embeddings
                print(f"Rotary Embs: Cos range=[{cos.min():.4f}, {cos.max():.4f}], Sin range=[{sin.min():.4f}, {sin.max():.4f}]")
                if torch.isnan(cos).any() or torch.isnan(sin).any():
                    print("CRITICAL: Rotary Embeddings produced NaNs!")
        
        # C. Layers
        enc_mask = torch.ones(1, enc_seq_len, device='cuda', dtype=torch.bool)
        
        for i, layer in enumerate(decoder.layers):
            # Debug: Check inputs to layer
            if torch.isnan(hidden_states).any():
                print(f"Layer {i} input contains NaNs!")
                break
            
            # For Layer 0, we want granular debug outputs
            if i == 0:
                # 1. Pre-SA Norm
                residual = hidden_states
                normed_hidden_states = layer.pre_self_attn_layernorm(hidden_states)
                np.save(f"{GOLDEN_DIR}/layer_0_norm_1.npy", normed_hidden_states.float().cpu().numpy())
                
                # 2. Self Attention
                # Replicate forward logic manually to capture output
                sa_out, _ = layer.self_attn(
                    hidden_states=normed_hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=None,
                    position_ids=None,
                    use_cache=False
                )
                np.save(f"{GOLDEN_DIR}/layer_0_attn_out.npy", sa_out.float().cpu().numpy())
                
                hidden_states = layer.post_self_attn_layernorm(sa_out)
                hidden_states = residual + hidden_states
                np.save(f"{GOLDEN_DIR}/layer_0_post_sa.npy", hidden_states.float().cpu().numpy())
                
                # 3. Cross Attention
                if encoder_hidden_states is not None:
                    residual = hidden_states
                    normed_hidden_states = layer.pre_cross_attn_layernorm(hidden_states)
                    ca_out, _ = layer.cross_attn(
                        hidden_states=normed_hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        attention_mask=enc_mask,
                        pm_decoder_position_ids=position_ids,
                        pm_encoder_position_ids=encoder_position_ids
                    )
                    np.save(f"{GOLDEN_DIR}/layer_0_cross_out.npy", ca_out.float().cpu().numpy())
                    hidden_states = layer.post_cross_attn_layernorm(ca_out)
                    hidden_states = residual + hidden_states
                
                # 4. MLP
                residual = hidden_states
                normed_hidden_states = layer.pre_feedforward_layernorm(hidden_states)
                np.save(f"{GOLDEN_DIR}/layer_0_norm_ff.npy", normed_hidden_states.float().cpu().numpy())
                
                ff_out = layer.mlp(normed_hidden_states)
                np.save(f"{GOLDEN_DIR}/layer_0_mlp_out.npy", ff_out.float().cpu().numpy())
                
                hidden_states = layer.post_feedforward_layernorm(ff_out)
                hidden_states = residual + hidden_states
                
            else:
                # Normal execution for other layers
                hidden_states = layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=enc_mask,
                    pm_decoder_position_ids=position_ids,
                    pm_encoder_position_ids=encoder_position_ids,
                    use_cache=False
                )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
            
            print(f"Layer {i} output: range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
            if torch.isnan(hidden_states).any():
                print(f"CRITICAL: Layer {i} produced NaNs!")
            
            np.save(f"{GOLDEN_DIR}/layer_{i}.npy", hidden_states.float().cpu().numpy())
            
        # D. Final Norm
        final_norm = getattr(decoder, "final_layer_norm", getattr(decoder, "norm", None))
        if final_norm:
            hidden_states = final_norm(hidden_states)
            
        print(f"Final output: range=[{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
        np.save(f"{GOLDEN_DIR}/output.npy", hidden_states.float().cpu().numpy())
        
    print(f"\nGolden outputs saved to {GOLDEN_DIR}")

if __name__ == "__main__":
    generate_golden()
