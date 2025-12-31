#!/usr/bin/env python3
"""
Export T5Gemma encoder to ONNX WITHOUT the problematic SDPA monkeypatch.

The original export script applied a monkeypatch that changed the model's behavior.
This script attempts to export the encoder without that modification.
"""
import os
_SAFE_TMPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_working"))
os.makedirs(_SAFE_TMPDIR, exist_ok=True)
os.environ["TMPDIR"] = _SAFE_TMPDIR
os.environ["TEMP"] = _SAFE_TMPDIR
os.environ["TMP"] = _SAFE_TMPDIR
import tempfile
tempfile.tempdir = _SAFE_TMPDIR

# Load env
with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import gc
import json
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SimpleEncoderWrapper(nn.Module):
    """Simple wrapper that just calls the encoder with PM-RoPE position IDs."""

    def __init__(self, encoder, progress_scale=2000.0):
        super().__init__()
        self.encoder = encoder
        self.progress_scale = progress_scale

    def forward(self, input_ids, attention_mask):
        # Build PM-RoPE position IDs
        x_lens = attention_mask.sum(dim=1)
        max_len = input_ids.shape[1]
        device = input_ids.device

        pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
        denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
        position_ids = pos / denom * self.progress_scale
        mask = pos < x_lens[:, None]
        position_ids = position_ids.masked_fill(~mask, 0.0)

        # Call encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs.last_hidden_state


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "./onnx_models_fp16_fixed"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model (NO monkeypatch)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "Aratako/T5Gemma-TTS-2b-2b",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    model.eval()

    # Get encoder and config
    progress_scale = getattr(model.config, "progress_scale", 2000.0)

    if hasattr(model, 'backbone'):
        encoder = model.backbone.model.encoder
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        encoder = model.model.encoder
    else:
        raise AttributeError("Cannot find encoder in model")

    logger.info(f"Progress scale: {progress_scale}")

    # Create wrapper
    wrapper = SimpleEncoderWrapper(encoder, progress_scale)
    wrapper.eval()

    # Prepare dummy inputs
    dummy_input_ids = torch.randint(0, 1000, (1, 64), device=device, dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 64), dtype=torch.long, device=device)

    # Test wrapper output
    with torch.no_grad():
        test_out = wrapper(dummy_input_ids, dummy_attention_mask)
        logger.info(f"Wrapper output shape: {test_out.shape}")
        logger.info(f"Wrapper output stats: mean={test_out.mean().item():.4f}, std={test_out.std().item():.4f}")

    # Try export with torch.onnx.dynamo_export (newer API)
    output_path = os.path.join(output_dir, "encoder.onnx")
    logger.info(f"Exporting to {output_path}...")

    # Move wrapper to CPU for export
    wrapper = wrapper.to("cpu")
    dummy_input_ids = dummy_input_ids.to("cpu")
    dummy_attention_mask = dummy_attention_mask.to("cpu")

    try:
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['encoder_hidden_states'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'},
                'encoder_hidden_states': {0: 'batch', 1: 'sequence'},
            }
        )
        logger.info(f"Export successful: {output_path}")

        # Verify the export
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])

        # Test with same input
        test_ids = torch.tensor([[32789, 1]], dtype=torch.long)  # "こんにちは" + EOS
        test_mask = torch.ones_like(test_ids)

        with torch.no_grad():
            wrapper = wrapper.to("cpu")
            pt_out = wrapper(test_ids, test_mask).numpy()

        onnx_out = session.run(None, {
            "input_ids": test_ids.numpy().astype(np.int64),
            "attention_mask": test_mask.numpy().astype(np.int64),
        })[0]

        cos_sim = np.dot(pt_out.flatten(), onnx_out.flatten()) / (
            np.linalg.norm(pt_out.flatten()) * np.linalg.norm(onnx_out.flatten()) + 1e-8
        )
        logger.info(f"\nVerification:")
        logger.info(f"  PyTorch: mean={pt_out.mean():.6f}, std={pt_out.std():.6f}")
        logger.info(f"  ONNX:    mean={onnx_out.mean():.6f}, std={onnx_out.std():.6f}")
        logger.info(f"  Cosine similarity: {cos_sim:.6f}")

        if cos_sim > 0.99:
            logger.info("  ✅ Export verified - outputs match!")
        else:
            logger.warning("  ⚠️ Export verification failed - outputs differ!")

    except Exception as e:
        logger.error(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

    # Save config
    conf_path = os.path.join(output_dir, "model_args.json")
    conf = model.config.to_dict() if hasattr(model.config, 'to_dict') else {}
    with open(conf_path, 'w') as f:
        json.dump(conf, f, indent=2, default=str)
    logger.info(f"Saved config to {conf_path}")


if __name__ == "__main__":
    main()
