import torch
import torch_tensorrt
import sys
import os

# Ensure we can import from data.tokenizer
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))

try:
    from data.tokenizer import AudioTokenizer
except ImportError:
    # Need to mock the imports or run in environment where xcodec2 is installed
    print("Please install xcodec2 to run this script: pip install xcodec2")
    sys.exit(1)

class VocoderWrapper(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.codec = tokenizer.codec
        
    def forward(self, codes):
        # codes: [Batch, 1, Seq]
        # xcodec2 expects [Batch, 1, Seq] (long)
        # But for tracing, inputs must be consistent.
        # Ensure we pass long tensors.
        recon = self.codec.decode_code(codes)
        return recon

def build(output_path):
    print("Loading AudioTokenizer (xcodec2)...")
    try:
        tokenizer = AudioTokenizer(backend="xcodec2", model_name="xcodec2", device="cuda")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    model = VocoderWrapper(tokenizer).eval().cuda()
    
    # Define Input Spec
    # Shape: [Batch, 1, Seq]
    # Seq is dynamic.
    min_shape = (1, 1, 10)
    opt_shape = (1, 1, 512)
    max_shape = (1, 1, 2048)
    
    # Create input example for tracing
    example_input = torch.randint(0, 1024, opt_shape, dtype=torch.long).cuda()
    
    print("Compiling with Torch-TensorRT...")
    
    # Note: XCodec2 uses Embedding layers which usually require int32/int64 inputs.
    # Torch-TRT handles this.
    
    # Compile
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch_tensorrt.Input(
            min_shape=min_shape,
            opt_shape=opt_shape,
            max_shape=max_shape,
            dtype=torch.long,
            name="codes"
        )],
        enabled_precisions={torch.float16}, # Enable FP16 for math
        workspace_size=2 << 30, # 2GB
        truncate_long_and_double=True
    )
    
    print("Saving TorchScript module...")
    torch.jit.save(trt_model, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("trt_weights", exist_ok=True)
    build("trt_weights/vocoder_trt.ts")
