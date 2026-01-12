"""
Test Embedding + Layer 0 + FinalNorm.
"""
import numpy as np
import torch
import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.layers import Embedding, RmsNorm
from tensorrt_llm.module import Module
from modeling import T5GemmaBlock

class TestLayer0(Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size, dtype='bfloat16')
        self.layer0 = T5GemmaBlock(config, 0)
        self.final_norm = RmsNorm(config.hidden_size, eps=config.rms_norm_eps, dtype='bfloat16')
    
    def forward(self, input_ids, encoder_hidden_states, position_ids, encoder_position_ids):
        # Cast input
        from tensorrt_llm.functional import cast
        encoder_hidden_states = cast(encoder_hidden_states, 'bfloat16')
        
        x = self.embed_tokens(input_ids)
        x = self.layer0(x, encoder_hidden_states=encoder_hidden_states, 
                        position_ids=position_ids, encoder_position_ids=encoder_position_ids)
        x = self.final_norm(x)
        return x

class T5Config:
    def __init__(self):
        self.vocab_size = 65541
        self.hidden_size = 2304
        self.d_kv = 256
        self.d_ff = 9216
        self.num_decoder_layers = 1
        self.num_attention_heads = 8
        self.num_kv_heads = 4
        self.rms_norm_eps = 1e-6
        self.rope_theta = 10000.0
        self.dtype = "bfloat16"
        self.dec_seq_len = 32
        self.enc_seq_len = 64

def run_test():
    print("=== Layer 0 Test ===")
    
    # Load weights
    weights = np.load("../trt_weights/weights.npz")
    
    config = T5Config()
    builder = Builder()
    network = builder.create_network()
    
    with net_guard(network):
        model = TestLayer0(config)
        
        # Mapping for this test model
        # embed_tokens.weight -> audio_embedding.weight
        # layer0.X -> layers.0.X
        # final_norm.weight -> final_norm.weight
        
        for name, param in model.named_parameters():
            if name == "embed_tokens.weight":
                val = weights["audio_embedding.weight"]
                print(f"Loaded {name} (dtype: {val.dtype})")
                param.value = torch.from_numpy(val).bfloat16()
            elif name.startswith("layer0."):
                mapped_key = name.replace("layer0.", "layers.0.")
                if mapped_key in weights:
                    val = weights[mapped_key]
                    print(f"Loaded {name} <- {mapped_key} (dtype: {val.dtype})")
                    param.value = torch.from_numpy(val).bfloat16()
                else:
                    print(f"[Warn] Missing mapped key for {name}: {mapped_key}")
            elif name == "final_norm.weight":
                val = weights["final_norm.weight"]
                print(f"Loaded {name} (dtype: {val.dtype})")
                param.value = torch.from_numpy(val).bfloat16()
            else:
                print(f"[Warn] Unhandled parameter: {name}")

        # Inputs
        BATCH=1
        DEC_SEQ=32
        ENC_SEQ=64
        input_ids = tensorrt_llm.Tensor(name='input_ids', dtype=tensorrt_llm.str_dtype_to_trt('int32'), shape=[BATCH, DEC_SEQ])
        enc_hidden = tensorrt_llm.Tensor(name='encoder_hidden_states', dtype=tensorrt_llm.str_dtype_to_trt('bfloat16'), shape=[BATCH, ENC_SEQ, config.hidden_size])
        pos_ids = tensorrt_llm.Tensor(name='position_ids', dtype=tensorrt_llm.str_dtype_to_trt('float32'), shape=[BATCH, DEC_SEQ])
        enc_pos_ids = tensorrt_llm.Tensor(name='encoder_position_ids', dtype=tensorrt_llm.str_dtype_to_trt('float32'), shape=[BATCH, ENC_SEQ])
        
        output = model(input_ids, enc_hidden, pos_ids, enc_pos_ids)
        output.mark_output('output', tensorrt_llm.str_dtype_to_trt('bfloat16'))

    # Build
    engine_config = builder.create_builder_config(
        name="test_layer0",
        precision="bfloat16",
        opt_level=0,
    )
    print("\nBuilding engine...")
    engine = builder.build_engine(network, engine_config)
    
    with open("test_layer0.engine", "wb") as f:
        f.write(engine)
    print("Saved to test_layer0.engine")
        
    # Run Inference
    import tensorrt as trt
    
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine_obj = runtime.deserialize_cuda_engine(engine)
    context = engine_obj.create_execution_context()
    
    # Deterministic inputs
    torch.manual_seed(42)
    input_ids_gpu = torch.randint(0, 1000, (BATCH, DEC_SEQ), dtype=torch.int32).cuda()
    enc_hidden_gpu = torch.randn(BATCH, ENC_SEQ, config.hidden_size, dtype=torch.bfloat16).cuda()
    pos_ids_gpu = torch.linspace(0, 1, DEC_SEQ).unsqueeze(0).cuda()
    enc_pos_ids_gpu = torch.linspace(0, 1, ENC_SEQ).unsqueeze(0).cuda()
    output_gpu = torch.empty(BATCH, DEC_SEQ, config.hidden_size, dtype=torch.bfloat16).cuda()
    
    context.set_tensor_address("input_ids", input_ids_gpu.data_ptr())
    context.set_tensor_address("encoder_hidden_states", enc_hidden_gpu.data_ptr())
    context.set_tensor_address("position_ids", pos_ids_gpu.data_ptr())
    context.set_tensor_address("encoder_position_ids", enc_pos_ids_gpu.data_ptr())
    context.set_tensor_address("output", output_gpu.data_ptr())
    
    print("\nRunning inference...")
    stream = torch.cuda.Stream()
    success = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    
    print(f"Success: {success}")
    print(f"Output shape: {output_gpu.shape}")
    print(f"Has NaN: {torch.isnan(output_gpu).any().item()}")
    
    if not torch.isnan(output_gpu).any():
        print(f"Range: [{output_gpu.min():.6f}, {output_gpu.max():.6f}]")
        print(f"Sample (first 5): {output_gpu[0,0,:5].tolist()}")
    else:
        nan_count = torch.isnan(output_gpu).sum().item()
        print(f"NaN count: {nan_count} / {output_gpu.numel()}")

if __name__ == "__main__":
    run_test()
