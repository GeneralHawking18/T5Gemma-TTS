import torch
import numpy as np
import os

def check_weights():
    # 1. Load PyTorch Weights
    pt_path = "weights/decoder_pmrope.bin"
    print(f"Loading PyTorch weights from {pt_path}...")
    pt_state = torch.load(pt_path, map_location="cpu")
    
    # 2. Load TRT Weights
    trt_path = "trt_weights/weights.npz"
    print(f"Loading TRT weights from {trt_path}...")
    trt_weights = np.load(trt_path)
    
    # 3. Compare Audio Embedding
    print("\n--- Audio Embedding Comparison ---")
    pt_key = "audio_embedding.0.weight"
    trt_key = "audio_embedding.weight"
    
    if pt_key in pt_state:
        pt_emb = pt_state[pt_key].float().numpy()
        print(f"PT {pt_key}: shape={pt_emb.shape}, mean={pt_emb.mean():.6f}, std={pt_emb.std():.6f}, min={pt_emb.min():.6f}, max={pt_emb.max():.6f}")
    else:
        print(f"PT key {pt_key} NOT FOUND")
        
    if trt_key in trt_weights:
        trt_emb = trt_weights[trt_key]
        print(f"TRT {trt_key}: shape={trt_emb.shape}, mean={trt_emb.mean():.6f}, std={trt_emb.std():.6f}, min={trt_emb.min():.6f}, max={trt_emb.max():.6f}")
    else:
        print(f"TRT key {trt_key} NOT FOUND")
        
    if pt_key in pt_state and trt_key in trt_weights:
        diff = np.abs(pt_emb - trt_emb).max()
        print(f"Max Difference: {diff}")
        
    # 4. Compare Layer 0 Q Weight
    print("\n--- Layer 0 Q Weight Comparison ---")
    pt_key = "backbone.model.decoder.layers.0.self_attn.q_proj.weight"
    trt_key = "layers.0.self_attn.q.weight"
    
    # Try alternate PT keys if prefix differs
    if pt_key not in pt_state:
        pt_key = "layers.0.self_attn.q_proj.weight"
    
    if pt_key in pt_state:
        pt_q = pt_state[pt_key].float().numpy()
        print(f"PT {pt_key}: shape={pt_q.shape}, mean={pt_q.mean():.6f}, std={pt_q.std():.6f}, min={pt_q.min():.6f}, max={pt_q.max():.6f}")
    else:
        print(f"PT key {pt_key} NOT FOUND (Available keys start with: {list(pt_state.keys())[:2]})")
        
    if trt_key in trt_weights:
        trt_q = trt_weights[trt_key]
        print(f"TRT {trt_key}: shape={trt_q.shape}, mean={trt_q.mean():.6f}, std={trt_q.std():.6f}, min={trt_q.min():.6f}, max={trt_q.max():.6f}")
    else:
        print(f"TRT key {trt_key} NOT FOUND")

    if pt_key in pt_state and trt_key in trt_weights:
        diff = np.abs(pt_q - trt_q).max()
        print(f"Max Difference: {diff}")

if __name__ == "__main__":
    check_weights()