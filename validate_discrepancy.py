import sys
import os
import torch
from transformers import PretrainedConfig

# Add the directory containing the new script to path
sys.path.append(os.path.join(os.getcwd(), "experimental_gemma3_onnx"))

try:
    from export_decoder_onnx import Gemma3Model
except ImportError:
    print("Could not import Gemma3Model from experimental_gemma3_onnx/export_decoder_onnx.py")
    sys.exit(1)

def validate():
    print("=== Validation Report: T5Gemma-TTS vs. Gemma3 ONNX Export ===\n")

    # 1. Define T5Gemma-like Config
    # We mock it to look like what the new script expects ("Gemma3ForCausalLM") 
    # just to see if it captures the specific logic.
    config = PretrainedConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=2048,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        partial_rotary_factor=1.0,
        layer_types=["sliding_attention", "global_attention"],
        sliding_window=512,
        query_pre_attn_scalar=1.0,
    )
    # The new script checks this:
    config.architectures = ["Gemma3ForCausalLM"] 
    config._name_or_path = "dummy_model"

    print("Attempting to build ONNX graph using the new script...")
    try:
        model = Gemma3Model(config, precision="fp32")
        model._build_inputs_and_outputs()
        
        # Manually trigger layer building to check structure
        # We can't run full build_model() because it tries to load weights from HF
        # So we inspect the _build_decoder_layer method logic by mocking.
        
        print("\n[Analysis of Generated Graph Inputs]")
        input_names = [inp.name for inp in model.model.graph.inputs]
        print(f"Graph Inputs: {input_names}")
        
        missing_inputs = []
        expected_t5gemma_inputs = [
            "encoder_hidden_states", 
            "encoder_attention_mask", 
            "pm_decoder_position_ids",
            "pm_encoder_position_ids"
        ]
        
        for exp in expected_t5gemma_inputs:
            if exp not in input_names:
                missing_inputs.append(exp)
        
        if missing_inputs:
            print(f"\nCRITICAL DISCREPANCY: Missing inputs required for T5Gemma-TTS: {missing_inputs}")
            print("The new script exports a standard Decoder-Only model, while T5Gemma-TTS is an Encoder-Decoder Hybrid (Cross-Attention).")
        else:
            print("Inputs seem to match (Unexpected).")

        print("\n[Analysis of Layer Structure]")
        # We can verify that _build_decoder_layer does not call anything related to cross-attention
        import inspect
        source_code = inspect.getsource(model._build_decoder_layer)
        
        if "CrossAttention" in source_code or "encoder_hidden_states" in source_code:
             print("Layer builder seems to reference cross-attention (Unexpected).")
        else:
             print("CRITICAL DISCREPANCY: The '_build_decoder_layer' method does not contain any Cross-Attention logic.")
             print("Standard Gemma 3 layers: Self-Attention -> MLP")
             print("T5Gemma-TTS layers:    Self-Attention -> Cross-Attention -> MLP")

    except Exception as e:
        print(f"Validation failed with error: {e}")

if __name__ == "__main__":
    validate()
