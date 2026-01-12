"""
Analyze differences between TensorRT and PyTorch intermediate outputs.
"""
import numpy as np
import os
import glob

def analyze():
    output_dir = "validation_outputs"
    
    # Find all layer files
    trt_files = sorted(glob.glob(f"{output_dir}/trt_layer_*.npy"))
    
    print(f"{'Layer':<10} | {'Shape':<20} | {'Max Diff':<12} | {'Mean Diff':<12} | {'Status'}")
    print("-" * 80)
    
    for trt_file in trt_files:
        basename = os.path.basename(trt_file)
        # trt_layer_X.npy -> pt_layer_X.npy
        pt_file = trt_file.replace("trt_", "pt_")
        
        layer_name = basename.replace("trt_", "").replace(".npy", "")
        
        if not os.path.exists(pt_file):
            print(f"{layer_name:<10} | {'MISSING PT':<20} | {'N/A':<12} | {'N/A':<12} | SKIP")
            continue
            
        trt_data = np.load(trt_file)
        pt_data = np.load(pt_file)
        
        if trt_data.shape != pt_data.shape:
            print(f"{layer_name:<10} | {str(trt_data.shape):<20} | {'SHAPE MIS':<12} | {'N/A':<12} | FAIL")
            continue
            
        diff = np.abs(trt_data - pt_data)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # Thresholds for bfloat16
        # bfloat16 epsilon is around 0.007, but accumulation can grow error.
        max_val = np.abs(pt_data).max()
        rel_diff = max_diff / (max_val + 1e-9) if max_val > 0 else 0.0

        status = "OK"
        if rel_diff > 0.05:
            status = "FAIL"
        elif rel_diff > 0.01:
            status = "WARN"
        elif max_diff > 5.0 and rel_diff > 0.005:
             status = "WARN"

        if np.isnan(max_diff): status = "NaN"
        
        print(f"{layer_name:<10} | {str(trt_data.shape):<20} | {max_diff:.6f}     | {mean_diff:.6f}     | {status}")

    # Check final output
    trt_out = f"{output_dir}/trt_output.npy"
    pt_out = f"{output_dir}/pt_output.npy"
    if os.path.exists(trt_out) and os.path.exists(pt_out):
        t = np.load(trt_out)
        p = np.load(pt_out)
        diff = np.abs(t - p)
        print("-" * 80)
        print(f"{'Final':<10} | {str(t.shape):<20} | {diff.max():.6f}     | {diff.mean():.6f}     | {'OK' if diff.max()<0.5 else 'WARN'}")

if __name__ == "__main__":
    analyze()
