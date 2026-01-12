import numpy as np
import os
import sys

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLDEN_DIR = f"{BASE_DIR}/golden_outputs"
TRT_DIR = f"{BASE_DIR}/validation_outputs"

def compare_offline():
    print(f"Comparing TRT outputs ({TRT_DIR}) vs Golden PyTorch ({GOLDEN_DIR})...\n")
    
    # Files to compare
    files = [
        ("embedding_out.npy", "trt_embedding_out.npy"),
        ("layer_0_norm_1.npy", "trt_layer_0_norm_1.npy"),
        ("layer_0_attn_out.npy", "trt_layer_0_attn_out.npy"),
        ("layer_0_post_sa.npy", "trt_layer_0_post_sa.npy"),
        ("layer_0_cross_out.npy", "trt_layer_0_cross_out.npy"),
        ("layer_0_norm_ff.npy", "trt_layer_0_norm_ff.npy"),
        ("layer_0_mlp_out.npy", "trt_layer_0_mlp_out.npy"),
        ("output.npy", "trt_output.npy")
    ]
    # Add layers
    for i in range(26):
        files.append((f"layer_{i}.npy", f"trt_layer_{i}.npy"))
        
    for pt_file, trt_file in files:
        pt_path = f"{GOLDEN_DIR}/{pt_file}"
        trt_path = f"{TRT_DIR}/{trt_file}"
        
        name = pt_file.replace(".npy", "")
        
        if not os.path.exists(pt_path):
            print(f"{name:<15}: SKIP (Golden missing)")
            continue
        if not os.path.exists(trt_path):
            print(f"{name:<15}: SKIP (TRT missing)")
            continue
            
        pt_data = np.load(pt_path)
        trt_data = np.load(trt_path)
        
        # Determine dtype for comparison
        pt_data = pt_data.astype(np.float32)
        trt_data = trt_data.astype(np.float32)
        
        # Check shapes
        if pt_data.shape != trt_data.shape:
            print(f"{name:<15}: SHAPE MISMATCH {pt_data.shape} vs {trt_data.shape}")
            continue
            
        # Check NaNs
        if np.isnan(pt_data).any() or np.isnan(trt_data).any():
            print(f"{name:<15}: NAN DETECTED (PT={np.isnan(pt_data).any()}, TRT={np.isnan(trt_data).any()})")
            continue
            
        # Diff
        diff = np.abs(pt_data - trt_data)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # Thresholds
        status = "OK"
        # Use relative tolerance for large values (BF16 precision is approx 1e-2 to 1e-3)
        # Avoid division by zero
        max_val = np.abs(pt_data).max()
        rel_diff = max_diff / (max_val + 1e-9) if max_val > 0 else 0.0

        if rel_diff > 0.05: # > 5% relative error -> FAIL
            status = "FAIL"
        elif rel_diff > 0.01: # > 1% relative error -> WARN
            status = "WARN"
        elif max_diff > 1.0 and rel_diff > 0.001: 
             # High absolute diff but small relative -> WARN (likely precision noise)
             status = "WARN"
        
        print(f"{name:<15}: MaxDiff={max_diff:.6f}, MeanDiff={mean_diff:.6f}, RelDiff={rel_diff:.6f}, PT_Range=[{pt_data.min():.2f}, {pt_data.max():.2f}], TRT_Range=[{trt_data.min():.2f}, {trt_data.max():.2f}] [{status}]")

if __name__ == "__main__":
    compare_offline()
