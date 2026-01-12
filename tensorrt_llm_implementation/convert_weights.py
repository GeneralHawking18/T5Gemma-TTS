import torch
import numpy as np
import os
import argparse
from transformers import AutoModelForSeq2SeqLM


def convert(model_name, output_dir):
    print(f"Loading PyTorch model: {model_name}...")

    if model_name.endswith(".bin"):
        print("Loading from binary file directly...")
        state_dict = torch.load(model_name, map_location="cpu")
    else:
        # Load model using transformers (handles download/cache)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="float16",  # Load in fp16 to save RAM
            device_map="cpu",  # Keep on CPU for conversion
        )
        state_dict = model.state_dict()

    weights = {}

    print("Converting weights...")
    for key, val in state_dict.items():
        # Convert to numpy (keep as float32 to avoid overflow/precision loss before TRT cast)
        val = val.to(torch.float32).cpu().numpy().astype(np.float32)

        # Mapping Logic
        # PyTorch: backbone.model.decoder.block.0.layer.0.SelfAttention.q.weight
        # TRT: layers.0.self_attn.q.weight

        new_key = key

        # Remove prefixes and identify component
        if "backbone.model.encoder." in new_key:
            new_key = new_key.replace("backbone.model.encoder.", "")
            component = "encoder"
        elif "backbone.model.decoder." in new_key:
            new_key = new_key.replace("backbone.model.decoder.", "")
            component = "decoder"
        elif "decoder." in new_key:  # Handle if loaded just decoder
            new_key = new_key.replace("decoder.", "")
            component = "decoder"
        else:
            component = "shared"  # Embeddings or other

        # T5Gemma structure mapping
        # Map decoder layers to TRT model format (layers.X without encoder/decoder prefix)
        if "layers." in new_key:
            parts = new_key.split(".")
            # parts[0] = layers, parts[1] = layer_idx
            if parts[0] == "layers":
                layer_idx = parts[1]
                suffix = ".".join(parts[2:])
            else:
                # Fallback
                layer_idx = "unknown"
                suffix = new_key

            # TRT model uses "layers.X" directly (no encoder/decoder prefix)
            new_prefix = f"layers.{layer_idx}"

            # RMSNorm layers (check BEFORE attn/mlp since names contain these substrings)
            # Pre-norms
            if "pre_self_attn_layernorm" in suffix:
                new_key = f"{new_prefix}.pre_sa_norm.weight"
            elif "pre_cross_attn_layernorm" in suffix:
                new_key = f"{new_prefix}.pre_ca_norm.weight"
            elif "pre_feedforward_layernorm" in suffix:
                new_key = f"{new_prefix}.pre_ff_norm.weight"
            # Post-norms
            elif "post_self_attn_layernorm" in suffix:
                new_key = f"{new_prefix}.post_sa_norm.weight"
            elif "post_cross_attn_layernorm" in suffix:
                new_key = f"{new_prefix}.post_ca_norm.weight"
            elif "post_feedforward_layernorm" in suffix:
                new_key = f"{new_prefix}.post_ff_norm.weight"

            # T5Gemma RMSNorm weights: Some are offsets (centered at 0), some are absolute (centered at 1 or more).
            # TRT-LLM RmsNorm expects full absolute weight.
            if "norm.weight" in new_key:
                mean_val = np.mean(val)
                if mean_val < 0.5:
                    # Offset weight (e.g. mean ~0 or -1) -> Add 1.0 to get absolute
                    val = val + 1.0
                    print(
                        f"  [Offset Norm] {new_key}: mean={mean_val:.4f} -> added 1.0"
                    )
                else:
                    # Already absolute weight (e.g. mean 2.7) -> Leave as is
                    print(
                        f"  [Absolute Norm] {new_key}: mean={mean_val:.4f} -> kept as is"
                    )

            # Self Attention
            elif "self_attn" in suffix:
                if "q_proj" in suffix:
                    new_key = f"{new_prefix}.self_attn.q.weight"
                elif "k_proj" in suffix:
                    new_key = f"{new_prefix}.self_attn.k.weight"
                elif "v_proj" in suffix:
                    new_key = f"{new_prefix}.self_attn.v.weight"
                elif "o_proj" in suffix:
                    new_key = f"{new_prefix}.self_attn.o.weight"

            # Cross Attention
            elif "cross_attn" in suffix:
                if "q_proj" in suffix:
                    new_key = f"{new_prefix}.cross_attn.q.weight"
                elif "k_proj" in suffix:
                    new_key = f"{new_prefix}.cross_attn.k.weight"
                elif "v_proj" in suffix:
                    new_key = f"{new_prefix}.cross_attn.v.weight"
                elif "o_proj" in suffix:
                    new_key = f"{new_prefix}.cross_attn.o.weight"

            # MLP
            elif "mlp" in suffix:
                if "gate_proj" in suffix:
                    new_key = f"{new_prefix}.mlp.gate.weight"
                elif "up_proj" in suffix:
                    new_key = f"{new_prefix}.mlp.up.weight"
                elif "down_proj" in suffix:
                    new_key = f"{new_prefix}.mlp.down.weight"

        elif "norm.weight" in new_key and "layers" not in key:
            # Final layer norm (backbone.model.decoder.norm.weight)
            new_key = "final_norm.weight"
            mean_val = np.mean(val)
            if mean_val < 0.5:
                val = val + 1.0
                print(f"  [Offset Norm] {new_key}: mean={mean_val:.4f} -> added 1.0")
            else:
                print(f"  [Absolute Norm] {new_key}: mean={mean_val:.4f} -> kept as is")

        elif "embed_tokens" in new_key:
            # Skip text embedding - we use audio_embedding for TTS
            print(f"Skipping text embedding: {key}")
            continue

        # Handle audio_embedding - this is what TTS decoder actually uses!
        elif "audio_embedding.0.weight" in key:
            new_key = "audio_embedding.weight"
            print(f"Mapped {key:<60} -> {new_key} (AUDIO EMBEDDING)")
            weights[new_key] = val
            continue

        # Handle predict_layer (LM Head)
        if "predict_layer" in key:
            # predict_layer is Sequential(Linear, GELU, Linear)
            # Map predict_layer.X -> lm_head.X
            new_key = key
            if "backbone.model.decoder." in new_key:
                new_key = new_key.replace("backbone.model.decoder.", "")
            elif "decoder." in new_key:
                new_key = new_key.replace("decoder.", "")

            new_key = new_key.replace("predict_layer", "lm_head")

            print(f"Mapped {key:<60} -> {new_key} (LM HEAD)")
            weights[new_key] = val
            continue

        # Skip keys we don't need (decoder_module duplicates)
        # Note: We want to keep post_*_layernorm weights, so don't skip "post_"
        if "decoder_module" in key:
            print(f"Skipping {key}")
            continue

        print(f"Mapped {key:<60} -> {new_key}")
        weights[new_key] = val

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "weights.npz")
    print(f"Saving {len(weights)} tensors to {out_path}...")
    np.savez(out_path, **weights)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Aratako/T5Gemma-TTS-2b-2b")
    parser.add_argument("--output_dir", type=str, default="trt_weights")
    args = parser.parse_args()
    convert(args.model_name, args.output_dir)
