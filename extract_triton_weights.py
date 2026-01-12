#!/usr/bin/env python3
"""
Extract predict_layer and audio_embedding weights from decoder checkpoint
for Triton Python backend deployment.

Usage:
    python extract_triton_weights.py \
        --weights weights/decoder_pmrope.bin \
        --output triton_model_repository/decoder_loop/1/weights
"""
import argparse
import json
import os
import shutil

import torch
import torch.nn as nn


def extract_weights(weights_path: str, output_dir: str):
    """Extract and save weights for Triton Python backend."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model args
    args_path = os.path.join(os.path.dirname(weights_path), "model_args.json")
    with open(args_path, "r") as f:
        model_args = json.load(f)

    # Copy model args
    output_args_path = os.path.join(output_dir, "model_args.json")
    shutil.copy(args_path, output_args_path)
    print(f"Copied model_args.json to {output_args_path}")

    # Load full checkpoint
    print(f"Loading checkpoint from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Model parameters
    hidden_size = 2304  # T5Gemma hidden size
    audio_vocab_size = model_args["audio_vocab_size"] + model_args["n_special"]
    print(f"Hidden size: {hidden_size}, Audio vocab size: {audio_vocab_size}")

    # Extract predict_layer weights
    # predict_layer is a Sequential: Linear -> GELU -> Linear
    predict_layer_keys = [k for k in state_dict.keys() if k.startswith("predict_layer.")]
    print(f"Found predict_layer keys: {predict_layer_keys}")

    # Reconstruct predict_layer
    predict_layer = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, audio_vocab_size),
    )

    # Map state dict keys
    predict_layer_state = {}
    for k, v in state_dict.items():
        if k.startswith("predict_layer.0."):
            # predict_layer.0.linear1.weight -> 0.weight
            new_key = k.replace("predict_layer.0.", "")
            predict_layer_state[new_key] = v
            print(f"  Mapped: {k} -> {new_key} (shape: {v.shape})")

    predict_layer.load_state_dict(predict_layer_state)
    predict_layer = predict_layer.to(dtype=torch.bfloat16).eval()

    # Save predict_layer
    predict_layer_path = os.path.join(output_dir, "predict_layer.pt")
    torch.save(predict_layer, predict_layer_path)
    print(f"Saved predict_layer to {predict_layer_path}")

    # Extract audio_embedding weights
    audio_embedding_keys = [k for k in state_dict.keys() if k.startswith("audio_embedding.")]
    print(f"Found audio_embedding keys: {audio_embedding_keys}")

    # audio_embedding is a ModuleList with single Embedding
    audio_embedding = nn.Embedding(audio_vocab_size, hidden_size)

    audio_embedding_state = {}
    for k, v in state_dict.items():
        if k.startswith("audio_embedding.0."):
            new_key = k.replace("audio_embedding.0.", "")
            audio_embedding_state[new_key] = v
            print(f"  Mapped: {k} -> {new_key} (shape: {v.shape})")

    audio_embedding.load_state_dict(audio_embedding_state)
    audio_embedding = audio_embedding.to(dtype=torch.bfloat16).eval()

    # Save audio_embedding
    audio_embedding_path = os.path.join(output_dir, "audio_embedding.pt")
    torch.save(audio_embedding, audio_embedding_path)
    print(f"Saved audio_embedding to {audio_embedding_path}")

    print(f"\nWeight extraction complete. Output directory: {output_dir}")
    print("Files created:")
    for f in os.listdir(output_dir):
        path = os.path.join(output_dir, f)
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  - {f} ({size:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Extract Triton weights from decoder checkpoint")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/decoder_pmrope.bin",
        help="Path to decoder checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="triton_model_repository/decoder_loop/1/weights",
        help="Output directory for extracted weights",
    )
    args = parser.parse_args()

    extract_weights(args.weights, args.output)


if __name__ == "__main__":
    main()
