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
        if hasattr(rotary, "inv_freq"):
            print(f"  Old inv_freq: range=[{rotary.inv_freq.min():.4e}, {rotary.inv_freq.max():.4e}]")
        
        # Deduce dim from inv_freq shape
        dim = rotary.inv_freq.shape[0] * 2
        print(f"  Duced dim={dim}")
        
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        rotary.inv_freq = inv_freq.to(device=rotary.inv_freq.device)
        rotary.cos_cached = None
        rotary.sin_cached = None
        
        if hasattr(rotary, "_set_cos_sin_cache"):
            rotary._set_cos_sin_cache(seq_len=2048, device=rotary.inv_freq.device, dtype=torch.float32)

    # Move to GPU with bfloat16
    model = model.to(dtype=torch.bfloat16, device='cuda')
    model.eval()
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
    
    # 1. Prepare Inputs
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size = 1
    dec_seq_len = 32
    enc_seq_len = 64
    hidden_size = 2304
    
    input_ids = torch.randint(0, 1000, (batch_size, dec_seq_len)).long().cuda()
    encoder_hidden_states = torch.randn(batch_size, enc_seq_len, hidden_size, dtype=torch.bfloat16).cuda()
    position_ids = torch.linspace(0, 1, dec_seq_len, dtype=torch.float32).unsqueeze(0).cuda()
    encoder_position_ids = torch.linspace(0, 1, enc_seq_len, dtype=torch.float32).unsqueeze(0).cuda()

    print("\nSaving inputs...")
    np.save(f"{GOLDEN_DIR}/input_ids.npy", input_ids.cpu().numpy())
    np.save(f"{GOLDEN_DIR}/encoder_hidden_states.npy", encoder_hidden_states.float().cpu().numpy())
    np.save(f"{GOLDEN_DIR}/position_ids.npy", position_ids.cpu().numpy())
    np.save(f"{GOLDEN_DIR}/encoder_position_ids.npy", encoder_position_ids.cpu().numpy())
    
    for f in ["input_ids.npy", "encoder_hidden_states.npy", "position_ids.npy", "encoder_position_ids.npy"]:
        shutil.copy(f"{GOLDEN_DIR}/{f}", f"{BASE_DIR}/validation_outputs/{f}")

    # 2. Run Forward Pass Layer-by-Layer
    print("\nRunning forward pass...")
    decoder = model.decoder_module
    
    with torch.no_grad():
        # A. Embedding
        audio_embedding = model.audio_embedding[0]
        hidden_states = audio_embedding(input_ids)
        np.save(f"{GOLDEN_DIR}/embedding_out.npy", hidden_states.float().cpu().numpy())
        
        # B. Rotary Embeddings
        sa_position_ids = torch.arange(dec_seq_len, dtype=torch.long, device='cuda').unsqueeze(0)
        position_embeddings = None
        if hasattr(decoder, "rotary_emb"):
            position_embeddings = decoder.rotary_emb(hidden_states, sa_position_ids)
        
        # C. Layers
        enc_mask = torch.ones(1, enc_seq_len, device='cuda', dtype=torch.bool)
        
        for i, layer in enumerate(decoder.layers):
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
