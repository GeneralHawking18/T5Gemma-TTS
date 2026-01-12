import torch
import sys
import os
import onnx

# Ensure we can import from data.tokenizer
sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath(".."))

try:
    from data.tokenizer import AudioTokenizer
except ImportError:
    print("Error: Could not import AudioTokenizer. Ensure xcodec2 and dependencies are installed.")
    sys.exit(1)

class VocoderWrapper(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.codec = tokenizer.codec
        
    def forward(self, codes):
        # codes: [Batch, 1, Seq] (Long)
        # xcodec2 decoding
        recon = self.codec.decode_code(codes)
        # Output: [Batch, 1, Audio_Len]
        return recon

def export_onnx(output_path):
    print("Loading AudioTokenizer (xcodec2)...")
    try:
        # Load on CPU for export to keep it simple and portable
        # Use default model or specific repo
        tokenizer = AudioTokenizer(backend="xcodec2", model_name="NandemoGHS/Anime-XCodec2-44.1kHz-v2", device="cpu")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    model = VocoderWrapper(tokenizer).eval()
    
    # Define Dummy Input
    # Shape: [Batch=1, Channels=1, Seq=128]
    dummy_input = torch.randint(0, 1024, (1, 1, 128), dtype=torch.long)
    
    print(f"Exporting to {output_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        input_names=['codes'],
        output_names=['audio'],
        dynamic_axes={
            'codes': {0: 'batch', 2: 'sequence'},
            'audio': {0: 'batch', 2: 'samples'}
        },
        dynamo=True
    )
    
    # Verification
    print("Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ Success! Saved to {output_path}")

if __name__ == "__main__":
    os.makedirs("onnx_models_fp16_fixed", exist_ok=True)
    export_onnx("onnx_models_fp16_fixed/vocoder.onnx")
