"""
Compare TensorRT outputs with local PyTorch T5GemmaVoiceModel.
Uses decoder_pmrope.bin weights for comparison.
"""

import torch
import numpy as np
import os
import sys
import json

# Add parent directory to path for model imports
# If running in Docker, project root is /app
if os.path.exists("/app"):
    PROJECT_ROOT = "/app"
else:
    PROJECT_ROOT = "/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS"

sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Paths
BASE_DIR = PROJECT_ROOT
WEIGHTS_DIR = f"{BASE_DIR}/weights"
TRT_WEIGHTS_PATH = f"{BASE_DIR}/trt_weights/weights.npz"
OUTPUT_DIR = f"{BASE_DIR}/validation_outputs"

def load_saved_inputs():
    """Load the inputs that were used for TRT inference."""
    print("Loading saved inputs...")
    inputs = {
        'input_ids': np.load(f"{OUTPUT_DIR}/input_ids.npy"),
        'encoder_hidden_states': np.load(f"{OUTPUT_DIR}/encoder_hidden_states.npy"),
        'position_ids': np.load(f"{OUTPUT_DIR}/position_ids.npy"),
        'encoder_position_ids': np.load(f"{OUTPUT_DIR}/encoder_position_ids.npy"),
    }
    trt_output = np.load(f"{OUTPUT_DIR}/trt_output.npy")
    return inputs, trt_output

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
    print("Patching from_pretrained to skip weight download...")
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    original_from_pretrained = AutoModelForSeq2SeqLM.from_pretrained
    
    def no_download_from_pretrained(model_name, **kwargs):
        print(f"  Intercepted download for {model_name}. Initializing from config (fast)...")
        config = AutoConfig.from_pretrained(model_name)
        # Respect dtype
        dtype = kwargs.get('torch_dtype', torch.float32)
        
        # Init on meta device to skip random init
        with torch.device("meta"):
            model = AutoModelForSeq2SeqLM.from_config(config)
            
        # Materialize on CPU (empty)
        model.to_empty(device="cpu")
        model.to(dtype=dtype)
        return model
    
    # Apply patch
    import transformers
    transformers.AutoModelForSeq2SeqLM.from_pretrained = no_download_from_pretrained

    # Create model
    try:
        print("Creating T5GemmaVoiceModel...")
        model = T5GemmaVoiceModel(args)
    finally:
        # Restore patch just in case
        transformers.AutoModelForSeq2SeqLM.from_pretrained = original_from_pretrained

    # Load weights
    decoder_weights_path = f"{WEIGHTS_DIR}/decoder_pmrope.bin"
    print(f"Loading decoder weights from {decoder_weights_path}...")
    state_dict = torch.load(decoder_weights_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights: {len(missing)} missing, {len(unexpected)} unexpected")
    if missing and len(missing) < 10:
        print(f"  Missing: {missing}")

    # Move to GPU with bfloat16 (same as inference)
    model = model.to(dtype=torch.bfloat16, device='cuda')
    model.eval()

    # Verify PM-RoPE is enabled
    pm_rope_enabled = getattr(model, "_pm_rope_enabled", False)
    print(f"PM-RoPE enabled: {pm_rope_enabled}")

    return model

def compare_outputs(name, pt_tensor, trt_tensor):
    """Compare PyTorch and TRT outputs."""
    pt_np = pt_tensor.detach().float().cpu().numpy()
    trt_np = trt_tensor.astype(np.float32)

    pt_nan = np.isnan(pt_np).any()
    trt_nan = np.isnan(trt_np).any()

    print(f"\n{name}:")
    print(f"  PyTorch: shape={pt_np.shape}, range=[{np.nanmin(pt_np):.4f}, {np.nanmax(pt_np):.4f}], NaN={pt_nan}")
    print(f"  TRT:     shape={trt_np.shape}, range=[{np.nanmin(trt_np):.4f}, {np.nanmax(trt_np):.4f}], NaN={trt_nan}")

    if pt_np.shape == trt_np.shape and not pt_nan and not trt_nan:
        diff = np.abs(pt_np - trt_np)
        print(f"  Diff: max={diff.max():.6e}, mean={diff.mean():.6e}")
        return diff.max()
    else:
        print(f"  Cannot compare: shape mismatch or NaN present")
        return float('inf')

def run_pytorch_layer_by_layer(model, inputs):
    """Run PyTorch decoder layer by layer and capture intermediate outputs."""
    print("\n" + "="*60)
    print("RUNNING PYTORCH DECODER LAYER BY LAYER")
    print("="*60)

    # Access the decoder module from T5GemmaVoiceModel
    decoder = model.decoder_module

    # Convert inputs to tensors
    # TRT used int32 for input_ids, but embedding expects int64
    input_ids = torch.from_numpy(inputs['input_ids']).long().cuda()
    encoder_hidden_states = torch.from_numpy(inputs['encoder_hidden_states']).to(dtype=torch.bfloat16, device='cuda')
    position_ids = torch.from_numpy(inputs['position_ids']).cuda()
    encoder_position_ids = torch.from_numpy(inputs['encoder_position_ids']).cuda()

    print(f"\nInputs:")
    print(f"  input_ids: {input_ids.shape}, dtype={input_ids.dtype}, range=[{input_ids.min()}, {input_ids.max()}]")
    print(f"  encoder_hidden_states: {encoder_hidden_states.shape}, dtype={encoder_hidden_states.dtype}")
    print(f"  position_ids: {position_ids.shape}, range=[{position_ids.min():.4f}, {position_ids.max():.4f}]")
    print(f"  encoder_position_ids: {encoder_position_ids.shape}")

    outputs = {}

    with torch.no_grad():
        # 1. Embedding - use audio_embedding for audio tokens
        # Note: T5GemmaVoiceModel uses audio_embedding, not decoder.embed_tokens
        audio_embedding = model.audio_embedding[0]  # First codebook embedding
        hidden_states = audio_embedding(input_ids)
        outputs['embedding'] = hidden_states.clone()
        print(f"\n1. After audio_embedding:")
        print(f"   Shape: {hidden_states.shape}")
        print(f"   Range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
        print(f"   NaN: {torch.isnan(hidden_states).any()}")
        
        # Compare embedding immediately
        trt_embed_path = f"{OUTPUT_DIR}/trt_embedding_out.npy"
        if os.path.exists(trt_embed_path):
            trt_embed = np.load(trt_embed_path)
            diff = compare_outputs("Embedding", hidden_states, trt_embed)
            if diff > 1e-5:
                print("CRITICAL: Embedding mismatch detected! Check weights and input_ids.")

        # Check decoder layers structure
        layers = decoder.layers
        print(f"\nDecoder has {len(layers)} layers")
        
        # Prepare rotary embeddings (Self-Attention uses standard RoPE with int positions)
        seq_len = hidden_states.shape[1]
        # Standard integer positions [0, 1, 2, ..., seq_len-1]
        sa_position_ids = torch.arange(seq_len, dtype=torch.long, device='cuda').unsqueeze(0)
        
        position_embeddings = None
        if hasattr(decoder, "rotary_emb"):
             position_embeddings = decoder.rotary_emb(hidden_states, sa_position_ids)
        else:
             print("Warning: decoder.rotary_emb not found!")

        # 2. Process each layer
        for layer_idx, layer in enumerate(layers):
            # Self-attention
            # Note: T5GemmaVoiceModel PMDecoderLayer already handles norms inside forward
            # We call layer(...) directly
            
            # Prepare kwargs for PM-RoPE
            # T5GemmaVoiceModel uses pm_decoder_position_ids kwarg
            
            # Encoder attention mask: all 1s for now (bool for SDPA)
            enc_mask = torch.ones(encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], device='cuda', dtype=torch.bool)
            
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=enc_mask,
                pm_decoder_position_ids=position_ids,
                pm_encoder_position_ids=encoder_position_ids,
                use_cache=False
            )
            # Depending on return type (tuple or tensor)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

            outputs[f'layer_{layer_idx}'] = hidden_states.clone()
            
            # Save intermediate output
            np.save(f"{OUTPUT_DIR}/pt_layer_{layer_idx}.npy", hidden_states.float().cpu().numpy())

            # Print info for first few and last layers
            if layer_idx < 3 or layer_idx >= len(layers) - 2:
                print(f"\n{layer_idx+2}. After layer {layer_idx}:")
                print(f"   Range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
                print(f"   NaN: {torch.isnan(hidden_states).any()}")
            elif layer_idx == 3:
                print(f"\n   ... (layers 3-{len(layers)-3}) ...")

        # 3. Final LayerNorm
        final_norm = None
        for norm_name in ['final_layer_norm', 'norm', 'layer_norm']:
            if hasattr(decoder, norm_name):
                final_norm = getattr(decoder, norm_name)
                break

        if final_norm:
            final_output = final_norm(hidden_states)
        else:
            final_output = hidden_states
            print("\nWARNING: No final layer norm found!")

        outputs['final_output'] = final_output.clone()
        np.save(f"{OUTPUT_DIR}/pt_output.npy", final_output.float().cpu().numpy())
        print(f"\n{len(layers)+2}. After final_layer_norm:")
        print(f"   Shape: {final_output.shape}")
        print(f"   Range: [{final_output.min():.4f}, {final_output.max():.4f}]")
        print(f"   NaN: {torch.isnan(final_output).any()}")

    return outputs

def compare_trt_weights_with_pytorch(model):
    """Compare TRT weights with PyTorch model weights."""
    print("\n" + "="*60)
    print("COMPARING WEIGHTS")
    print("="*60)

    trt_weights = np.load(TRT_WEIGHTS_PATH)
    decoder = model.decoder_module

    # For T5GemmaVoiceModel, the audio_embedding is separate
    audio_embedding = model.audio_embedding[0]

    # Key mappings: TRT key -> (PyTorch weight, description)
    # Note: Need to figure out the correct mapping based on actual layer structure
    print("\nTRT weight keys available:")
    for key in sorted(trt_weights.files)[:20]:
        print(f"  {key}: shape={trt_weights[key].shape}")
    if len(trt_weights.files) > 20:
        print(f"  ... and {len(trt_weights.files) - 20} more")

    print("\nPyTorch decoder structure:")
    print(f"  decoder type: {type(decoder)}")
    if hasattr(decoder, 'layers'):
        print(f"  num_layers: {len(decoder.layers)}")
        layer0 = decoder.layers[0]
        print(f"  layer0 type: {type(layer0)}")
        print(f"  layer0 attrs: {[a for a in dir(layer0) if 'weight' in a.lower() or 'attn' in a.lower() or 'norm' in a.lower() or 'mlp' in a.lower()][:15]}")

    # Try to compare embedding weights
    print("\n--- Embedding comparison ---")
    if 'embed_tokens.weight' in trt_weights:
        trt_embed = trt_weights['embed_tokens.weight']
        pt_embed = audio_embedding.weight.data.float().cpu().numpy()
        print(f"TRT embed: {trt_embed.shape}, dtype={trt_embed.dtype}")
        print(f"PT audio_embed: {pt_embed.shape}, dtype={pt_embed.dtype}")

        if trt_embed.shape == pt_embed.shape:
            diff = np.abs(trt_embed.astype(np.float32) - pt_embed.astype(np.float32)).max()
            print(f"Max diff: {diff:.6e}")
        else:
            print("*** CRITICAL MISMATCH! ***")
            print("TRT uses embed_tokens (256000 vocab) - this is TEXT embedding!")
            print("PyTorch uses audio_embedding (65541 vocab) - this is AUDIO embedding!")
            print("The TRT model is using WRONG EMBEDDING TABLE!")

    return []

def run_full_decoder_forward(model, inputs):
    """Run full decoder forward pass (simpler approach)."""
    print("\n" + "="*60)
    print("RUNNING FULL DECODER FORWARD")
    print("="*60)

    decoder = model.decoder_module
    audio_embedding = model.audio_embedding[0]

    # Convert inputs
    input_ids = torch.from_numpy(inputs['input_ids']).long().cuda()
    encoder_hidden_states = torch.from_numpy(inputs['encoder_hidden_states']).to(dtype=torch.bfloat16, device='cuda')
    position_ids = torch.from_numpy(inputs['position_ids']).cuda()
    encoder_position_ids = torch.from_numpy(inputs['encoder_position_ids']).cuda()

    with torch.no_grad():
        # Embed input tokens
        inputs_embeds = audio_embedding(input_ids)
        print(f"inputs_embeds: {inputs_embeds.shape}, range=[{inputs_embeds.min():.4f}, {inputs_embeds.max():.4f}]")

        # Run full decoder
        # Check what arguments the decoder accepts
        print(f"\nDecoder forward signature check...")

        try:
            outputs = decoder(
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=torch.ones(1, encoder_hidden_states.shape[1], device='cuda').long(),
                pm_decoder_position_ids=position_ids,
                pm_encoder_position_ids=encoder_position_ids,
                use_cache=False,
            )
            hidden_states = outputs.last_hidden_state
            print(f"\nDecoder output (with PM-RoPE position IDs):")
            print(f"  Shape: {hidden_states.shape}")
            print(f"  Range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
            print(f"  NaN: {torch.isnan(hidden_states).any()}")
            return hidden_states
        except TypeError as e:
            print(f"Error with PM-RoPE position IDs: {e}")

            # Try without PM-RoPE specific args
            try:
                outputs = decoder(
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    use_cache=False,
                )
                hidden_states = outputs.last_hidden_state
                print(f"\nDecoder output (without PM-RoPE):")
                print(f"  Shape: {hidden_states.shape}")
                print(f"  Range: [{hidden_states.min():.4f}, {hidden_states.max():.4f}]")
                print(f"  NaN: {torch.isnan(hidden_states).any()}")
                return hidden_states
            except Exception as e2:
                print(f"Error: {e2}")
                return None

def main():
    # Check if TRT outputs exist
    if not os.path.exists(f"{OUTPUT_DIR}/trt_output.npy"):
        print("ERROR: TRT outputs not found! Run save_trt_outputs.py first.")
        return

    # Load saved inputs and TRT output
    inputs, trt_output = load_saved_inputs()

    print(f"\nTRT Output stats:")
    print(f"  Shape: {trt_output.shape}")
    print(f"  Range: [{np.nanmin(trt_output):.4f}, {np.nanmax(trt_output):.4f}]")
    print(f"  NaN: {np.isnan(trt_output).any()}")

    # Load local model
    model = load_local_model()

    # Compare weights
    compare_trt_weights_with_pytorch(model)

    # Run layer by layer to debug
    layer_outputs = run_pytorch_layer_by_layer(model, inputs)
    pt_output = layer_outputs['final_output']

    # Compare outputs
    if pt_output is not None:
        print("\n" + "="*60)
        print("FINAL OUTPUT COMPARISON")
        print("="*60)
        compare_outputs("Final Output", pt_output, trt_output)

        # Sample comparison
        print("\nSample values (first 5 elements of position 0):")
        pt_sample = pt_output[0, 0, :5].float().cpu().tolist()
        trt_sample = trt_output[0, 0, :5].tolist()
        print(f"  PyTorch: {pt_sample}")
        print(f"  TRT:     {trt_sample}")

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
