import torch
import numpy as np
import argparse
from transformers import AutoModelForSeq2SeqLM

def verify(model_name, weights_path):
    print(f"Loading Original PyTorch Model: {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="float16",
        device_map="cpu"
    )
    pt_state = model.state_dict()
    
    print(f"Loading Exported Weights: {weights_path}...")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    trt_weights = np.load(weights_path)
    
    print("\nStarting Layer-by-Layer Verification...")
    print("="*80)
    print(f"{ 'TRT Name':<50} | { 'PyTorch Name':<50} | {'Status'}")
    print("-" * 110)
    
    # Define the mapping logic again (reverse of convert_weights) or iterate TRT keys
    # Iterating TRT keys is better to ensure every exported weight has a match
    
    passed = 0
    failed = 0
    
    # Helper to find matching PT key
    # We essentially re-implement the mapping logic but in reverse/search mode
    # Since convert_weights was explicit, we can't easily invert it programmatically without
    # recreating the map. Instead, we'll verify the critical layers.
    
    # 1. Embeddings
    check_layer(pt_state, trt_weights, "shared.weight", "embed_tokens.weight") # T5 usually shares
    # T5Gemma might separate them. Let's check convert_weights output.
    # It mapped "backbone.model.decoder.embed_tokens.weight" -> "embed_tokens.weight"
    
    check_layer(pt_state, trt_weights, "backbone.model.decoder.embed_tokens.weight", "embed_tokens.weight")
    check_layer(pt_state, trt_weights, "backbone.model.decoder.norm.weight", "final_norm.weight")
    
    # 2. Check a few random layers (Start, Middle, End)
    # Layer 0 Self Attn
    check_layer(pt_state, trt_weights, "backbone.model.decoder.layers.0.self_attn.q_proj.weight", "layers.0.self_attn.q.weight")
    check_layer(pt_state, trt_weights, "backbone.model.decoder.layers.0.self_attn.o_proj.weight", "layers.0.self_attn.o.weight")
    
    # Layer 12 Cross Attn
    check_layer(pt_state, trt_weights, "backbone.model.decoder.layers.12.cross_attn.k_proj.weight", "layers.12.cross_attn.k.weight")
    
    # Layer 23 MLP
    check_layer(pt_state, trt_weights, "backbone.model.decoder.layers.23.mlp.up_proj.weight", "layers.23.wi.weight")
    
    # 3. Comprehensive Scan
    # We iterate all keys in weights.npz and try to reconstruct the PT key source
    # This detects if we exported garbage.
    
    print("\nComprehensive Scan of All Exported Weights:")
    for trt_key in trt_weights.files:
        trt_val = trt_weights[trt_key]
        
        # We don't know exact PT source without the map logic, 
        # but we can sanity check values (not NaN, not zero)
        if np.isnan(trt_val).any():
            print(f"[FAIL] {trt_key} contains NaNs!")
            failed += 1
        elif np.all(trt_val == 0):
            print(f"[WARN] {trt_key} is all zeros!")
        else:
            # OK
            pass
            
    print("-" * 110)
    print(f"Verification Complete.")

def check_layer(pt_state, trt_weights, pt_key, trt_key):
    if pt_key not in pt_state:
        print(f"{trt_key:<50} | {pt_key:<50} | SKIPPED (PT Key Missing)")
        return

    if trt_key not in trt_weights:
        print(f"{trt_key:<50} | {pt_key:<50} | FAIL (TRT Key Missing)")
        return

    pt_val = pt_state[pt_key].float().cpu().numpy()
    trt_val = trt_weights[trt_key].astype(np.float32) # Cast back up for comparison
    
    # Shape check
    if pt_val.shape != trt_val.shape:
        print(f"{trt_key:<50} | {pt_key:<50} | FAIL (Shape Mismatch: {trt_val.shape} vs {pt_val.shape})")
        return

    # Value check
    diff = np.abs(pt_val - trt_val).max()
    # FP16 precision tolerance ~1e-3
    if diff < 1e-3:
        print(f"{trt_key:<50} | {pt_key:<50} | PASS (Diff: {diff:.6f})")
    else:
        print(f"{trt_key:<50} | {pt_key:<50} | FAIL (Diff: {diff:.6f})")

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Aratako/T5Gemma-TTS-2b-2b")
    parser.add_argument("--weights_path", type=str, default="trt_weights/weights.npz")
    args = parser.parse_args()
    
    verify(args.model_name, args.weights_path)
