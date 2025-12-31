"""
ONNX Encoder-Only Inference Script
Loads only the encoder ONNX model and runs inference with dummy inputs.
"""
import os
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv

load_dotenv()

class ONNXEncoderInference:
    def __init__(self, onnx_dir: str = "onnx_models_fp16"):
        """
        Initialize the ONNX encoder inference.
        
        Args:
            onnx_dir: Directory containing encoder.onnx and external data files
        """
        self.onnx_dir = onnx_dir
        self.encoder_path = os.path.join(onnx_dir, "encoder.onnx")
        
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder ONNX not found at {self.encoder_path}")
        
        print(f"Loading ONNX encoder from {self.encoder_path}...")
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Use CPU provider for now (can add CUDA later)
        providers = ['CPUExecutionProvider']
        
        # Load encoder session
        self.encoder_session = ort.InferenceSession(
            self.encoder_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output info
        self.encoder_inputs = {inp.name: inp for inp in self.encoder_session.get_inputs()}
        self.encoder_outputs = {out.name: out for out in self.encoder_session.get_outputs()}
        
        print(f"Encoder loaded successfully!")
        print(f"Encoder inputs: {list(self.encoder_inputs.keys())}")
        print(f"Encoder outputs: {list(self.encoder_outputs.keys())}")
        
        # Print input shapes
        for name, inp in self.encoder_inputs.items():
            print(f"  Input '{name}': shape={inp.shape}, dtype={inp.type}")
    
    def run_encoder(self, input_ids: np.ndarray, attention_mask: np.ndarray = None, position_ids: np.ndarray = None):
        """
        Run encoder forward pass.
        
        Args:
            input_ids: Token IDs, shape [batch, seq_len]
            attention_mask: Attention mask, shape [batch, seq_len] (optional)
            position_ids: Position IDs, shape [batch, seq_len] (optional)
            
        Returns:
            encoder_hidden_states: shape [batch, seq_len, hidden_size]
        """
        # Prepare inputs - build dict based on what the model expects
        onnx_inputs = {}
        
        # Always add input_ids
        if "input_ids" in self.encoder_inputs:
            onnx_inputs["input_ids"] = input_ids.astype(np.int64)
        
        # Add attention_mask if expected
        if "attention_mask" in self.encoder_inputs:
            if attention_mask is None:
                attention_mask = np.ones_like(input_ids, dtype=np.int64)
            onnx_inputs["attention_mask"] = attention_mask.astype(np.int64)
        
        # Add position_ids if expected
        if "position_ids" in self.encoder_inputs:
            if position_ids is None:
                seq_len = input_ids.shape[1]
                position_ids = np.arange(seq_len, dtype=np.float32)[None, :]
                position_ids = np.tile(position_ids, (input_ids.shape[0], 1))
            onnx_inputs["position_ids"] = position_ids.astype(np.float32)
        
        # Run encoder
        outputs = self.encoder_session.run(None, onnx_inputs)
        
        return outputs[0]  # Return encoder hidden states


def test_encoder():
    """Test the ONNX encoder with dummy inputs."""
    print("=" * 60)
    print("Testing ONNX Encoder-Only Inference")
    print("=" * 60)
    
    # Initialize encoder
    encoder = ONNXEncoderInference(onnx_dir="onnx_models_fp16")
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 20
    
    # Random token IDs (assuming vocab size around 256000 for T5Gemma)
    input_ids = np.random.randint(0, 1000, size=(batch_size, seq_len), dtype=np.int64)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    
    print(f"\nRunning encoder with input_ids shape: {input_ids.shape}")
    
    try:
        hidden_states = encoder.run_encoder(input_ids, attention_mask)
        print(f"\nEncoder output shape: {hidden_states.shape}")
        print(f"Encoder output dtype: {hidden_states.dtype}")
        print(f"Output sample (first 5 values): {hidden_states[0, 0, :5]}")
        print("\n✅ ONNX Encoder inference successful!")
    except Exception as e:
        print(f"\n❌ Encoder inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_encoder()
