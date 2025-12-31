"""
Export decoder weights WITH PM-RoPE layers from T5GemmaVoiceModel.

This exports everything needed for hybrid inference EXCEPT the encoder:
- Decoder with PM-RoPE cross-attention layers
- Audio embeddings
- Predict layers
- Config/args
"""
import torch
import os
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

# Add current directory to path for models import
import sys
sys.path.insert(0, os.getcwd())

from models.t5gemma import T5GemmaVoiceModel


def create_args_from_config(config):
    """Create args namespace from HuggingFace config."""
    class Args:
        pass
    
    args = Args()
    
    # Core model settings
    args.t5gemma_model_name = getattr(config, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2")
    args.n_codebooks = getattr(config, "n_codebooks", 1)
    args.audio_vocab_size = getattr(config, "audio_vocab_size", 65536)
    args.n_special = getattr(config, "n_special", 5)
    
    # Special tokens
    args.empty_token = getattr(config, "empty_token", 65536)
    args.eog = getattr(config, "eog", 65537)
    args.eos = getattr(config, "eos", 65539)
    args.audio_pad_token = getattr(config, "audio_pad_token", 65540)
    
    # PM-RoPE settings
    args.use_pm_rope = getattr(config, "use_pm_rope", 1)
    args.progress_scale = getattr(config, "progress_scale", 2000.0)
    
    # Audio settings
    args.codec_audio_sr = getattr(config, "codec_audio_sr", 44100)
    args.encodec_sr = getattr(config, "encodec_sr", 50)
    args.audio_max_length = getattr(config, "audio_max_length", 30)
    
    # Other settings
    args.text_input_type = "text"
    args.text_vocab_size = 0
    args.precision = "bfloat16"
    args.attn_implementation = "eager"
    args.prune_text_modules = 1  # Drop lm_head to save memory
    args.freeze_t5gemma = 0
    args.use_lora = 0
    args.t5_gradient_checkpointing = 0
    args.eog_weight = 1.0
    args.special_first = getattr(config, "special_first", 0)
    
    # Separators
    args.x_sep_token = getattr(config, "x_sep_token", None)
    args.y_sep_token = getattr(config, "y_sep_token", None)
    
    return args


def export_decoder_pmrope(model_name, output_dir, use_bf16=True):
    """
    Export decoder + PM-RoPE + audio components from T5GemmaVoiceModel.
    
    Args:
        model_name: HuggingFace model name (e.g., "Aratako/T5Gemma-TTS-2b-2b")
        output_dir: Directory to save weights
        use_bf16: Use BFloat16 (recommended for stability)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"[Export] Loading model from {model_name}...")
    print(f"[Export] Using dtype: {dtype}")
    
    # Load config first
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Create args from config
    args = create_args_from_config(config)
    
    # Save args for later loading
    args_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    args_path = os.path.join(output_dir, "model_args.json")
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    print(f"[Export] Saved model args to {args_path}")
    
    # Load T5GemmaVoiceModel (this enables PM-RoPE layers!)
    print(f"[Export] Creating T5GemmaVoiceModel with PM-RoPE enabled...")
    model = T5GemmaVoiceModel(args)
    model = model.to(dtype=dtype)
    model.eval()
    
    # Load pretrained weights
    print(f"[Export] Loading pretrained weights...")
    from transformers import AutoModelForSeq2SeqLM
    pretrained = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    # Get pretrained state dict and remap keys
    pretrained_state = pretrained.state_dict()
    
    # Remap keys from AutoModelForSeq2SeqLM format to T5GemmaVoiceModel format
    # AutoModel uses: backbone.model.encoder.*, backbone.model.decoder.*
    # T5GemmaVoiceModel expects the same, but carefully_load_state_dict expects exact match
    print(f"[Export] Loading weights directly (bypassing carefully_load_state_dict)...")
    
    # Load backbone weights directly into model.backbone
    missing, unexpected = model.backbone.load_state_dict(
        {k.replace("backbone.", ""): v for k, v in pretrained_state.items() if k.startswith("backbone.")},
        strict=False
    )
    print(f"[Export] Backbone: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    
    # Load audio embedding and predict layer if present
    audio_embed_keys = [k for k in pretrained_state if "audio_embedding" in k]
    predict_keys = [k for k in pretrained_state if "predict_layer" in k]
    print(f"[Export] Audio embedding keys: {len(audio_embed_keys)}, Predict layer keys: {len(predict_keys)}")
    
    if audio_embed_keys or predict_keys:
        # Load remaining weights
        remaining = {k: v for k, v in pretrained_state.items() if not k.startswith("backbone.")}
        model.load_state_dict(remaining, strict=False)
    
    # Verify PM-RoPE is enabled
    pm_rope_enabled = getattr(model, "_pm_rope_enabled", False)
    print(f"[Export] PM-RoPE enabled: {pm_rope_enabled}")
    
    # Get full state dict
    full_state = model.state_dict()
    
    # Filter: keep everything EXCEPT encoder
    decoder_state = {}
    encoder_dropped = 0
    
    for k, v in full_state.items():
        # Drop encoder keys
        if any(pattern in k for pattern in [
            "backbone.model.encoder",
            "backbone.encoder", 
            "encoder_module",
        ]):
            encoder_dropped += 1
            continue
        if k.startswith("encoder."):
            encoder_dropped += 1
            continue
            
        # Keep everything else (decoder, embeddings, predict layers)
        decoder_state[k] = v
    
    print(f"[Export] Kept {len(decoder_state)} keys")
    print(f"[Export] Dropped {encoder_dropped} encoder keys")
    
    # Check for PM-RoPE specific keys
    pm_rope_keys = [k for k in decoder_state.keys() if "rotary_emb" in k or "decoder_rotary" in k or "encoder_rotary" in k]
    print(f"[Export] PM-RoPE related keys: {len(pm_rope_keys)}")
    if pm_rope_keys[:5]:
        print(f"[Export] Sample PM-RoPE keys: {pm_rope_keys[:5]}")
    
    # Save weights
    weights_path = os.path.join(output_dir, "decoder_pmrope.bin")
    print(f"[Export] Saving to {weights_path}...")
    torch.save(decoder_state, weights_path)
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    config.save_pretrained(output_dir)
    print(f"[Export] Saved config to {config_path}")
    
    # Print summary
    total_params = sum(p.numel() for p in decoder_state.values())
    print(f"\n[Export] ✅ Export complete!")
    print(f"[Export] Total parameters: {total_params:,} ({total_params * 2 / 1e9:.2f} GB in BF16)")
    print(f"[Export] Files saved to: {output_dir}/")
    print(f"         - decoder_pmrope.bin")
    print(f"         - model_args.json")
    print(f"         - config.json")
    
    # Cleanup
    del model, pretrained
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return weights_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export decoder weights with PM-RoPE")
    parser.add_argument("--model_name", type=str, default="Aratako/T5Gemma-TTS-2b-2b",
                        help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="weights",
                        help="Output directory for weights")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 instead of BFloat16")
    args = parser.parse_args()
    
    export_decoder_pmrope(
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_bf16=not args.fp32
    )
