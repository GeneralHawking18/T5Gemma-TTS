import numpy as np
import argparse

def list_keys(weights_path):
    print(f"Loading {weights_path}...")
    weights = np.load(weights_path)
    print("Keys in weights.npz:")
    for key in sorted(weights.files)[:20]: # Print first 20
        print(f"{key}: {val.shape}")
    print("...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, default="trt_weights/weights.npz")
    args = parser.parse_args()
    list_keys(args.weights_path)
