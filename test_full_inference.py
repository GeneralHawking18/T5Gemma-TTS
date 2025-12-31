"""
Test script for full TTS inference with audio output.
Uses hybrid ONNX encoder + PyTorch decoder (T5GemmaVoiceModel with PM-RoPE) + XCodec2 vocoder.
Matches the output of inference_4bit_gpu.py

Now refactored to use the synthesize() method from HybridT5GemmaTTS.
"""
import os
import time
import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def run_full_inference(
    target_text="podcastをsubscribeしています。",
    top_k=30,
    top_p=0.9,
    min_p=0.0,
    temperature=0.7,
    stop_repetition=3,
    target_duration=None,
    output_dir="outputs",
    seed=1,
):
    """
    Run full TTS inference matching inference_4bit_gpu.py output.

    Args:
        target_text: Text to synthesize
        top_k: Top-k sampling (default: 30, same as inference_4bit_gpu)
        top_p: Top-p sampling (default: 0.9, same as inference_4bit_gpu)
        min_p: Min-p sampling (default: 0.0)
        temperature: Sampling temperature (default: 0.7, same as inference_4bit_gpu)
        stop_repetition: Stop repetition threshold (default: 3)
        target_duration: Target duration in seconds (auto-estimate if None)
        output_dir: Output directory for audio files
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    total_start = time.time()

    # Import hybrid TTS with PM-RoPE support (uses T5GemmaVoiceModel)
    from inference_hybrid_complete import HybridT5GemmaTTS
    from inference_tts_utils import save_audio

    # =========================================================
    # 1. Load Hybrid TTS Model
    # =========================================================
    print("=" * 60)
    print("Loading Hybrid TTS Model (T5GemmaVoiceModel with PM-RoPE)")
    print("=" * 60)

    load_start = time.time()
    tts = HybridT5GemmaTTS(
        onnx_dir="onnx_models_fp16",
        decoder_weights="weights/decoder_pmrope.bin",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # =========================================================
    # 2. Synthesize Audio (uses built-in synthesize() method)
    # =========================================================
    print(f"\n" + "=" * 60)
    print(f"Synthesizing audio for: '{target_text}'")
    print("=" * 60)
    print(f"Parameters: top_k={top_k}, top_p={top_p}, min_p={min_p}, temp={temperature}")

    synth_start = time.time()
    sample_rate, gen_audio = tts.synthesize(
        text=target_text,
        language=None,  # Auto-detect
        target_duration=target_duration,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        temperature=temperature,
        stop_repetition=stop_repetition,
    )
    synth_time = time.time() - synth_start

    print(f"\nSynthesis time: {synth_time:.2f}s")
    print(f"Audio waveform shape: {gen_audio.shape}")

    # =========================================================
    # 3. Save Audio
    # =========================================================
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "generated_hybrid.wav")

        # Convert to torch for save_audio (if numpy)
        if isinstance(gen_audio, np.ndarray):
            gen_audio = torch.from_numpy(gen_audio)

        # Use save_audio from inference_tts_utils
        save_audio(output_path, gen_audio, sample_rate)

        # Calculate audio stats
        max_abs = torch.max(gen_audio.abs()).item() if isinstance(gen_audio, torch.Tensor) else np.abs(gen_audio).max()
        rms_val = torch.sqrt((gen_audio ** 2).mean()).item() if isinstance(gen_audio, torch.Tensor) else np.sqrt((gen_audio ** 2).mean())
        print(f"[Info] Generated audio stats -> max_abs: {max_abs:.6f}, rms: {rms_val:.6f}")

        audio_duration = gen_audio.shape[-1] / sample_rate

        # =========================================================
        # 4. Summary
        # =========================================================
        total_time = time.time() - total_start

        print("\n" + "=" * 60)
        print("Inference Complete!")
        print("=" * 60)
        print(f"Audio saved to: {output_path}")
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"\nTiming breakdown:")
        print(f"  - Model loading: {load_time:.2f}s")
        print(f"  - Synthesis (text -> audio): {synth_time:.2f}s")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Real-time factor: {total_time / audio_duration:.2f}x")

        return output_path

    except Exception as e:
        print(f"Audio saving failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import fire
    fire.Fire(run_full_inference)
