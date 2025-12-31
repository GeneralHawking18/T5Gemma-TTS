#!/usr/bin/env python3
"""
T5Gemma-TTS Setup Module.

Download và cache models từ HuggingFace.
Chạy 1 lần để download trước khi start API.

Usage:
    python setup.py
"""

import os
import gc

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch
torch.set_grad_enabled(False)

from dotenv import load_dotenv
load_dotenv()


class ModelSetup:
    """Download và cache T5Gemma-TTS models."""
    
    # Pre-quantized 4-bit model
    DEFAULT_MODEL_ID = "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit"
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.config = None
        self.tokenizer_name = None
    
    def _print_memory(self, label: str = ""):
        try:
            import psutil
            mem = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            print(f"[Memory] {label}: {mem:.1f} MB")
        except ImportError:
            pass
    
    def download_model(self) -> "ModelSetup":
        """Download và cache model từ HuggingFace."""
        print(f"[1/3] Downloading model from {self.model_id}...")
        self._print_memory()
        
        from transformers import AutoModelForSeq2SeqLM
        
        # Download (sẽ được cache tự động bởi HuggingFace)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.config = model.config
        
        print("[OK] Model downloaded và cached!")
        self._print_memory()
        
        # Free memory
        del model
        gc.collect()
        return self
    
    def download_tokenizer(self) -> "ModelSetup":
        """Download text tokenizer."""
        print("[2/3] Downloading tokenizer...")
        
        from transformers import AutoTokenizer
        
        self.tokenizer_name = (
            getattr(self.config, "text_tokenizer_name", None) or
            getattr(self.config, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2")
        )
        
        AutoTokenizer.from_pretrained(self.tokenizer_name)
        print(f"[OK] Tokenizer: {self.tokenizer_name}")
        return self
    
    def download_audio_tokenizer(self) -> "ModelSetup":
        """Download XCodec2 audio tokenizer."""
        print("[3/3] Downloading XCodec2...")
        
        try:
            from data.tokenizer import AudioTokenizer
            
            AudioTokenizer(
                backend="xcodec2",
                model_name=getattr(self.config, "xcodec2_model_name", None),
                device="cpu",
            )
            print("[OK] XCodec2 downloaded")
        except Exception as e:
            print(f"[Warning] XCodec2 download failed: {e}")
        
        return self
    
    def run(self) -> None:
        """Run full setup."""
        print("=" * 60)
        print("T5Gemma-TTS Setup (Pre-quantized 4-bit)")
        print("=" * 60)
        
        self.download_model()
        self.download_tokenizer()
        self.download_audio_tokenizer()
        
        print("\n" + "=" * 60)
        print("Setup complete! Models đã được cache.")
        print("Next: uvicorn api:app --host 0.0.0.0 --port 8000")
        print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup T5Gemma-TTS")
    parser.add_argument("--model-id", default=ModelSetup.DEFAULT_MODEL_ID)
    args = parser.parse_args()
    
    setup = ModelSetup(model_id=args.model_id)
    setup.run()


if __name__ == "__main__":
    main()
