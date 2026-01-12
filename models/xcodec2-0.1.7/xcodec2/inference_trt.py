"""
TensorRT Inference for XCodec2 Vocoder
"""
import tensorrt as trt
import numpy as np
import torch
import soundfile as sf
from typing import Union, Optional
import os


class XCodec2TRTInference:
    """TensorRT inference wrapper for XCodec2 vocoder."""

    def __init__(self, engine_path: str, device: int = 0):
        """
        Initialize TensorRT inference engine.

        Args:
            engine_path: Path to TensorRT engine file (.engine)
            device: CUDA device index
        """
        self.device = device
        torch.cuda.set_device(device)

        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        print(f"Loading TensorRT engine from {engine_path}...")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        print(f"Engine loaded successfully!")
        print(f"  - Input: vq_code [batch, 1, seq_length]")
        print(f"  - Output: recon_audio [batch, 1, audio_length]")

    def decode(self, vq_code: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Decode VQ codes to audio waveform.

        Args:
            vq_code: VQ codes with shape [batch, 1, seq_length] or [batch, seq_length]
                     dtype should be int32 or int64

        Returns:
            audio: Reconstructed audio waveform [batch, audio_length]
        """
        # Convert to numpy if torch tensor
        if isinstance(vq_code, torch.Tensor):
            vq_code = vq_code.cpu().numpy()

        # Ensure correct shape [batch, 1, seq_length]
        if vq_code.ndim == 2:
            vq_code = vq_code[:, np.newaxis, :]

        # Ensure int32
        vq_code = vq_code.astype(np.int32)

        batch_size, _, seq_length = vq_code.shape

        # Set input shape for dynamic axes
        self.context.set_input_shape('vq_code', vq_code.shape)

        # Get output shape
        output_shape = tuple(self.context.get_tensor_shape('recon_audio'))

        # Allocate GPU tensors
        vq_code_gpu = torch.from_numpy(vq_code).cuda(self.device)
        output_gpu = torch.empty(output_shape, dtype=torch.float32, device=f'cuda:{self.device}')

        # Set tensor addresses
        self.context.set_tensor_address('vq_code', vq_code_gpu.data_ptr())
        self.context.set_tensor_address('recon_audio', output_gpu.data_ptr())

        # Execute inference
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # Return output
        output = output_gpu.cpu().numpy()

        # Squeeze middle dimension: [batch, 1, audio_length] -> [batch, audio_length]
        if output.shape[1] == 1:
            output = output.squeeze(1)

        return output

    def decode_to_file(
        self,
        vq_code: Union[np.ndarray, torch.Tensor],
        output_path: str,
        sample_rate: int = 44100
    ) -> str:
        """
        Decode VQ codes and save to audio file.

        Args:
            vq_code: VQ codes
            output_path: Output audio file path
            sample_rate: Audio sample rate (default 44100 for XCodec2)

        Returns:
            output_path: Path to saved audio file
        """
        audio = self.decode(vq_code)

        # If batch size is 1, squeeze batch dimension
        if audio.shape[0] == 1:
            audio = audio.squeeze(0)

        sf.write(output_path, audio, sample_rate)
        print(f"Audio saved to {output_path} ({len(audio)/sample_rate:.2f}s)")

        return output_path


def main():
    """Example usage."""
    import argparse
    import time


    parser = argparse.ArgumentParser(description='XCodec2 TensorRT Inference')
    parser.add_argument('--engine', type=str, default='../vocoder.engine',
                        help='Path to TensorRT engine file')
    parser.add_argument('--input', type=str, default='../vq_code_cache.npy',
                        help='Path to input VQ codes (.npy)')
    parser.add_argument('--output', type=str, default='output.wav',
                        help='Output audio file path')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Audio sample rate')
    parser.add_argument('--device', type=int, default=0,
                        help='CUDA device index')

    args = parser.parse_args()

    # Initialize inference engine
    inference = XCodec2TRTInference(args.engine, device=args.device)

    # Load input
    print(f"Loading VQ codes from {args.input}...")
    vq_code = np.load(args.input)
    print(f"  Shape: {vq_code.shape}")

    # Run inference
    # print("Running TensorRT inference...")
    # audio = inference.decode(vq_code)
    # print(f"  Output shape: {audio.shape}")
    # print(f"  Output range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Save output
    start = time.time()
    inference.decode_to_file(vq_code, args.output, args.sample_rate)

    print(f"Done with inference time {time.time()-start}")


if __name__ == '__main__':
    main()
