import numpy as np
import os
import argparse

def compare_npz(onnx_path, trt_path, verbose=False):
    print(f"Comparing ONNX ({onnx_path}) and TensorRT ({trt_path}) outputs...")

    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found: {onnx_path}")
        return
    if not os.path.exists(trt_path):
        print(f"Error: TRT file not found: {trt_path}")
        return

    onnx_data = np.load(onnx_path)
    trt_data = np.load(trt_path)

    onnx_keys = set(onnx_data.files)
    trt_keys = set(trt_data.files)

    common_keys = sorted(list(onnx_keys.intersection(trt_keys)))
    only_onnx = onnx_keys - trt_keys
    only_trt = trt_keys - onnx_keys

    if not common_keys:
        print("Error: No common keys found between ONNX and TRT files.")
        print(f"ONNX keys: {onnx_keys}")
        print(f"TRT keys: {trt_keys}")
        return

    print(f"Found {len(common_keys)} common keys.")
    if only_onnx:
        print(f"Keys only in ONNX: {only_onnx}")
    if only_trt:
        print(f"Keys only in TRT: {only_trt}")

    print("\n" + "="*100)
    print(f"{ 'Key':<30} | { 'Shape':<20} | { 'Max Diff':<12} | { 'Mean Diff':<12} | Status")
    print("-" * 100)

    for key in common_keys:
        o = onnx_data[key].astype(np.float32)
        t = trt_data[key].astype(np.float32)

        if o.shape != t.shape:
            print(f"{key:<30} | {str(o.shape)} vs {str(t.shape)} | {'SHAPE MIS':<12} | {'N/A':<12} | FAIL")
            continue

        diff = np.abs(o - t)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # Calculate relative difference
        max_val = np.max(np.abs(o))
        rel_diff = max_diff / (max_val + 1e-9) if max_val > 0 else 0.0

        status = "OK"
        if np.isnan(max_diff):
            status = "NaN"
        elif rel_diff > 0.05:
            status = "FAIL"
        elif rel_diff > 0.01:
            status = "WARN"
        elif max_diff > 1.0 and rel_diff > 0.001:
            status = "WARN"

        print(f"{key:<30} | {str(o.shape):<20} | {max_diff:.6f}     | {mean_diff:.6f}     | {status}")
        
        if verbose and status != "OK":
            print(f"  -> ONNX range: [{o.min():.4f}, {o.max():.4f}]")
            print(f"  -> TRT range:  [{t.min():.4f}, {t.max():.4f}]")
            print(f"  -> Rel Diff:   {rel_diff:.6f}")

    print("="*100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare ONNX and TensorRT .npz outputs")
    parser.add_argument("--onnx", type=str, default="onnx_output.npz", help="Path to ONNX .npz file")
    parser.add_argument("--trt", type=str, default="trt_output.npz", help="Path to TRT .npz file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if files exist with provided names or common defaults
    onnx_file = args.onnx
    if not os.path.exists(onnx_file) and os.path.exists("onnx.npz"):
        onnx_file = "onnx.npz"
        
    trt_file = args.trt
    if not os.path.exists(trt_file) and os.path.exists("trt.npz"):
        trt_file = "trt.npz"

    compare_npz(onnx_file, trt_file, args.verbose)
