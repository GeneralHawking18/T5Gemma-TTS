import argparse
import os
import torch
import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.logger import logger
import numpy as np

# Import our custom model
from modeling import T5GemmaDecoderTRT, T5GemmaDecoderWithLMHead


class T5Config:
    def __init__(self):
        # T5Gemma decoder config for TTS
        # Correct config from Aratako/T5Gemma-TTS-2b-2b model:
        # - hidden_size=2304, head_dim=256, num_heads=8, num_kv_heads=4
        # - q_inner_dim = 8*256 = 2048, kv_inner_dim = 4*256 = 1024
        self.vocab_size = 65541  # Audio token vocab size (65536 + 5 special tokens)
        self.hidden_size = 2304
        self.d_kv = 256  # head_dim
        self.d_ff = 9216
        self.num_decoder_layers = 26
        self.num_attention_heads = 8  # Q heads
        self.num_kv_heads = 4  # K/V heads (GQA)
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0
        self.query_pre_attn_scalar = 256.0
        self.dtype = "bfloat16"
        # Fixed sequence lengths for RoPE (set during build)
        self.dec_seq_len = 32
        self.enc_seq_len = 64


def build(weights_path, output_dir):
    logger.set_level("info")

    # 1. Config
    config = T5Config()

    # 2. Builder
    builder = Builder()
    network = builder.create_network()

    # 3. Define Network
    with net_guard(network):
        model = T5GemmaDecoderWithLMHead(config)

        # Load weights with key mapping
        # weights.npz uses "audio_embedding.weight" for audio tokens (vocab=65541)
        # The T5GemmaDecoderWithLMHead wraps decoder, so params have "decoder." prefix
        # but weights.npz has keys without that prefix (e.g., "layers.0..." not "decoder.layers.0...")

        # Static mappings for special keys
        static_mapping = {
            "decoder.embed_tokens.weight": "audio_embedding.weight",
            "lm_head_0.weight": "lm_head.0.0.weight",
            "lm_head_0.bias": "lm_head.0.0.bias",
            "lm_head_2.weight": "lm_head.0.2.weight",
            "lm_head_2.bias": "lm_head.0.2.bias",
        }

        loaded_keys = set()
        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}...")
            weights = np.load(weights_path)
            available_keys = set(weights.files)
            print(f"Available keys in weights.npz: {len(available_keys)}")

            for name, param in model.named_parameters():
                # Try static mapping first
                if name in static_mapping:
                    weight_key = static_mapping[name]
                # Strip "decoder." prefix for decoder weights
                elif name.startswith("decoder."):
                    weight_key = name[len("decoder.") :]
                else:
                    weight_key = name

                if weight_key in weights:
                    param.value = torch.from_numpy(weights[weight_key]).bfloat16()
                    loaded_keys.add(name)
                    print(f"Loaded {name} <- {weight_key}")
                else:
                    print(f"[Warn] Missing weight: {name} (tried {weight_key})")
        else:
            print(f"ERROR: Weights file not found at {weights_path}")
            print(f"Current directory: {os.getcwd()}")
            print(
                f"Contents of ..: {os.listdir('..') if os.path.exists('..') else 'N/A'}"
            )

        # Check critical weights (decoder.embed_tokens because we wrapped it)
        if "decoder.embed_tokens.weight" not in loaded_keys:
            print(
                f"Warning: decoder.embed_tokens.weight not explicitly loaded. Check logs."
            )

        # Inputs - Use dynamic shapes (-1) for batch and sequence dimensions
        # BATCH = 1 (fixed for now to simplify, but could be dynamic)
        # DEC_SEQ = dynamic
        # ENC_SEQ = dynamic

        input_ids = tensorrt_llm.Tensor(
            name="input_ids",
            dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            shape=[-1, -1],
        )
        enc_hidden = tensorrt_llm.Tensor(
            name="encoder_hidden_states",
            dtype=tensorrt_llm.str_dtype_to_trt("bfloat16"),
            shape=[-1, -1, config.hidden_size],
        )
        pos_ids = tensorrt_llm.Tensor(
            name="position_ids",
            dtype=tensorrt_llm.str_dtype_to_trt("float32"),
            shape=[-1, -1],
        )
        enc_pos_ids = tensorrt_llm.Tensor(
            name="encoder_position_ids",
            dtype=tensorrt_llm.str_dtype_to_trt("float32"),
            shape=[-1, -1],
        )
        enc_mask = tensorrt_llm.Tensor(
            name="encoder_attention_mask",
            dtype=tensorrt_llm.str_dtype_to_trt("int32"),
            shape=[-1, -1],
        )

        # Forward
        logits = model(
            input_ids, enc_hidden, pos_ids, enc_pos_ids, encoder_attention_mask=enc_mask
        )

        # Mark Output (The final logits)
        logits.mark_output("logits", tensorrt_llm.str_dtype_to_trt("bfloat16"))

    # 4. Build Engine
    engine_config = builder.create_builder_config(
        name="t5gemma_decoder",
        precision="bfloat16",
        timing_cache="timing.cache",
        opt_level=0,
    )

    # Add Optimization Profile for dynamic shapes
    import tensorrt as trt

    trt_builder = trt.Builder(logger.trt_logger)
    profile = trt_builder.create_optimization_profile()

    # input_ids: [batch, dec_seq]
    profile.set_shape("input_ids", (1, 1), (1, 32), (1, 2048))
    # encoder_hidden_states: [batch, enc_seq, hidden]
    profile.set_shape(
        "encoder_hidden_states",
        (1, 1, config.hidden_size),
        (1, 128, config.hidden_size),
        (1, 1024, config.hidden_size),
    )
    # position_ids: [batch, dec_seq]
    profile.set_shape("position_ids", (1, 1), (1, 32), (1, 2048))
    # encoder_position_ids: [batch, enc_seq]
    profile.set_shape("encoder_position_ids", (1, 1), (1, 128), (1, 1024))
    # encoder_attention_mask: [batch, enc_seq]
    profile.set_shape("encoder_attention_mask", (1, 1), (1, 128), (1, 1024))

    engine_config.trt_builder_config.add_optimization_profile(profile)

    print("Building Engine...")
    engine = builder.build_engine(network, engine_config)

    os.makedirs(output_dir, exist_ok=True)
    engine_path = os.path.join(output_dir, "t5gemma_decoder_with_lm_head.engine")
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"Engine saved to {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights_path", type=str, default="../trt_weights/weights.npz"
    )
    parser.add_argument("--output_dir", type=str, default="engine_output")
    args = parser.parse_args()

    build(args.weights_path, args.output_dir)
