"""
Minimal test: Build ONLY embedding layer and verify output is not NaN.
This isolates whether the issue is in weight loading or computation.
"""
import numpy as np
import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.network import net_guard
from tensorrt_llm.layers import Embedding
from tensorrt_llm.module import Module

class MinimalEmbedding(Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed_tokens = Embedding(vocab_size, hidden_size, dtype='bfloat16')
    
    def forward(self, input_ids):
        return self.embed_tokens(input_ids)

def test_embedding_only():
    print("=== Minimal Embedding Test ===")
    
    # Load weights
    weights = np.load("../trt_weights/weights.npz")
    audio_emb = weights["audio_embedding.weight"]
    print(f"Loaded audio_embedding.weight: shape={audio_emb.shape}, dtype={audio_emb.dtype}")
    print(f"  Has NaN: {np.isnan(audio_emb).any()}")
    print(f"  Range: [{audio_emb.min():.6f}, {audio_emb.max():.6f}]")
    
    # Build minimal model
    builder = Builder()
    network = builder.create_network()
    
    VOCAB_SIZE = 65541
    HIDDEN_SIZE = 2304
    BATCH = 1
    SEQ = 32
    
    with net_guard(network):
        model = MinimalEmbedding(VOCAB_SIZE, HIDDEN_SIZE)
        
        # Check what parameters exist
        print("\n=== Model Parameters ===")
        for name, param in model.named_parameters():
            print(f"  {name}: type={type(param)}")
        
        # Load embedding weights
        for name, param in model.named_parameters():
            if "embed_tokens" in name:
                print(f"\nLoading {name} from audio_embedding.weight...")
                param.value = audio_emb
                print(f"  Loaded! Shape: {audio_emb.shape}")
        
        # Define input
        input_ids = tensorrt_llm.Tensor(
            name='input_ids', 
            dtype=tensorrt_llm.str_dtype_to_trt('int32'), 
            shape=[BATCH, SEQ]
        )
        
        # Forward
        output = model(input_ids)
        output.mark_output('output', tensorrt_llm.str_dtype_to_trt('bfloat16'))
    
    # Build engine
    engine_config = builder.create_builder_config(
        name="test_embedding",
        precision="bfloat16",
        opt_level=0,
    )
    
    print("\nBuilding engine...")
    engine = builder.build_engine(network, engine_config)
    
    # Save engine
    with open("test_embedding.engine", "wb") as f:
        f.write(engine)
    print("Saved to test_embedding.engine")
    
    # Now run inference
    import tensorrt as trt
    import torch
    
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine_obj = runtime.deserialize_cuda_engine(engine)
    context = engine_obj.create_execution_context()
    
    # Input: token IDs 0, 1, 2, ...
    input_ids_gpu = torch.arange(SEQ, dtype=torch.int32).unsqueeze(0).cuda().contiguous()
    output_gpu = torch.empty(BATCH, SEQ, HIDDEN_SIZE, dtype=torch.bfloat16).cuda().contiguous()
    
    context.set_tensor_address("input_ids", input_ids_gpu.data_ptr())
    context.set_tensor_address("output", output_gpu.data_ptr())
    
    stream = torch.cuda.Stream()
    success = context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    
    print("\n=== Inference Result ===")
    print(f"Success: {success}")
    print(f"Output shape: {output_gpu.shape}")
    print(f"Has NaN: {torch.isnan(output_gpu).any().item()}")
    print(f"Has Inf: {torch.isinf(output_gpu).any().item()}")
    
    if not torch.isnan(output_gpu).any():
        print(f"Range: [{output_gpu.min():.6f}, {output_gpu.max():.6f}]")
        print(f"First token output[:5]: {output_gpu[0, 0, :5].tolist()}")
        
        # Compare with numpy lookup
        expected = audio_emb[0, :5]  # Token 0
        print(f"Expected (numpy):      {expected.tolist()}")
    else:
        nan_count = torch.isnan(output_gpu).sum().item()
        print(f"NaN count: {nan_count} / {output_gpu.numel()}")

if __name__ == "__main__":
    test_embedding_only()
