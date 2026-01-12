import os
import sys
import torch
import numpy as np
import tensorrt_llm
from tqdm import tqdm

# Add the main implementation to path to reuse modeling.py and build.py logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tensorrt_llm_implementation")))

try:
    from modeling import T5GemmaDecoderTRT
    from build import T5Config
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure tensorrt_llm_implementation folder is structured correctly.")
    sys.exit(1)

def build_layer0(weights_path, output_dir):
    print("Initializing isolated Layer 0 Build...")
    
    # 1. Force a 1-layer configuration
    config = T5Config()
    config.num_decoder_layers = 1
    # Check if config has query_pre_attn_scalar, otherwise default to 256.0
    if not hasattr(config, "query_pre_attn_scalar"):
        config.query_pre_attn_scalar = 256.0
    
    # 2. Setup Builder
    builder = tensorrt_llm.builder.Builder()
    network = builder.create_network()
    
    with tensorrt_llm.builder.net_guard(network):
        model = T5GemmaDecoderTRT(config)
        
        # 3. Load weights with Progress Tracking
        if not os.path.exists(weights_path):
            print(f"Weights not found: {weights_path}")
            sys.exit(1)
            
        weights = np.load(weights_path)
        
        print(f"Mapping weights into Layer 0 network...")
        loaded_count = 0
        params = list(model.named_parameters())
        for name, param in tqdm(params, desc="Weight Loading"):
            weight_key = name
            # Handle standard mapping
            if name == "embed_tokens.weight":
                weight_key = "audio_embedding.weight"
            
            if weight_key in weights:
                param.value = torch.from_numpy(weights[weight_key]).bfloat16()
                loaded_count += 1
            else:
                # Silently skip weights for layers we pruned (layers 1-25)
                pass
        
        print(f"Successfully loaded {loaded_count} parameters for Layer 0.")

        # 4. Define Inputs
        BATCH, DEC_SEQ, ENC_SEQ = 1, 32, 64
        input_ids = tensorrt_llm.Tensor(name='input_ids', dtype=tensorrt_llm.str_dtype_to_trt('int32'), shape=[BATCH, DEC_SEQ])
        enc_hidden = tensorrt_llm.Tensor(name='encoder_hidden_states', dtype=tensorrt_llm.str_dtype_to_trt('bfloat16'), shape=[BATCH, ENC_SEQ, config.hidden_size])
        pos_ids = tensorrt_llm.Tensor(name='position_ids', dtype=tensorrt_llm.str_dtype_to_trt('float32'), shape=[BATCH, DEC_SEQ])
        enc_pos_ids = tensorrt_llm.Tensor(name='encoder_position_ids', dtype=tensorrt_llm.str_dtype_to_trt('float32'), shape=[BATCH, ENC_SEQ])
        enc_mask = tensorrt_llm.Tensor(name='encoder_attention_mask', dtype=tensorrt_llm.str_dtype_to_trt('int32'), shape=[BATCH, ENC_SEQ])
        
        # 5. Build Graph
        output, hidden_states = model(input_ids, enc_hidden, pos_ids, enc_pos_ids, encoder_attention_mask=enc_mask)
        output.mark_output('output', tensorrt_llm.str_dtype_to_trt('bfloat16'))

    # 6. Build Engine
    engine_config = builder.create_builder_config(
        name="l0_debug",
        precision="bfloat16",
        opt_level=0 
    )
    
    print("Building Engine (Layer 0 isolated)...")
    engine = builder.build_engine(network, engine_config)
    
    os.makedirs(output_dir, exist_ok=True)
    engine_path = os.path.join(output_dir, "l0_debug.engine")
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"Engine saved to {engine_path}")

if __name__ == "__main__":
    # Correct relative path to trt_weights
    build_layer0("../trt_weights/weights.npz", "./engine")
