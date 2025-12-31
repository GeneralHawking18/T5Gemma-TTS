#!/usr/bin/env python3
"""
Export T5Gemma encoder to ONNX with CORRECTED monkeypatch.

Uses the original EncoderWrapper from onnx_modules.py but with
the corrected (non-causal) monkeypatch.
"""
import os
_SAFE_TMPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp_working"))
os.makedirs(_SAFE_TMPDIR, exist_ok=True)
os.environ["TMPDIR"] = _SAFE_TMPDIR
os.environ["TEMP"] = _SAFE_TMPDIR
os.environ["TMP"] = _SAFE_TMPDIR

with open(".env") as f:
    for line in f:
        if "=" in line:
            k, v = line.strip().split("=", 1)
            os.environ[k] = v

import gc
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import transformers.masking_utils

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CORRECTED MONKEYPATCH - Full bidirectional attention (matches original)
# =============================================================================
def corrected_sdpa_mask(batch_size, cache_position, kv_length, kv_offset=0,
                        mask_function=None, attention_mask=None, **kwargs):
    """
    Corrected SDPA mask - full bidirectional attention.
    This produces output IDENTICAL to the original T5Gemma encoder.
    """
    device = cache_position.device
    q_length = cache_position.shape[0]

    # Full attention - all positions can attend to all positions
    causal_mask = torch.ones(
        (batch_size, 1, q_length, kv_length),
        dtype=torch.bool,
        device=device
    )

    if attention_mask is not None:
        padding_mask = attention_mask.to(torch.bool)[:, None, None, :]
        causal_mask = causal_mask & padding_mask

    return causal_mask

# Apply corrected monkeypatch
transformers.masking_utils.sdpa_mask = corrected_sdpa_mask
transformers.masking_utils.sdpa_mask_recent_torch = corrected_sdpa_mask
transformers.masking_utils.sdpa_mask_older_torch = corrected_sdpa_mask
if hasattr(transformers.masking_utils, "ALL_MASK_ATTENTION_FUNCTIONS"):
    mapping = transformers.masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    if "sdpa" in mapping:
        mapping["sdpa"] = corrected_sdpa_mask

logger.info("Applied CORRECTED monkeypatch (bidirectional attention)")

# Import the original wrapper
from onnx_modules import EncoderWrapper


def main():
    output_dir = "./onnx_models_fp16_fixed"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "Aratako/T5Gemma-TTS-2b-2b",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    model.eval()
    model.to(dtype=torch.float16)
    cfg = model.config

    # Setup compatibility attributes
    if not hasattr(model, "args"):
        model.args = model.config
    if not hasattr(model, "encoder_module"):
        if hasattr(model, "backbone"):
            model.encoder_module = model.backbone.model.encoder
        elif hasattr(model, "model"):
            model.encoder_module = model.model.encoder
    if not hasattr(model, "text_input_type"):
        model.text_input_type = getattr(cfg, "text_input_type", "text")
    if not hasattr(model, "progress_scale"):
        model.progress_scale = getattr(cfg, "progress_scale", 2000.0)
    if not hasattr(model, "text_embedding"):
        if hasattr(model, "backbone"):
            model.text_embedding = model.backbone.model.encoder.embed_tokens
        elif hasattr(model.model, "encoder"):
            model.text_embedding = model.model.encoder.embed_tokens
    if not hasattr(model, "text_dropout"):
        model.text_dropout = nn.Identity()

    logger.info(f"Progress scale: {model.progress_scale}")

    # Create wrapper
    wrapper = EncoderWrapper(model)
    wrapper.eval()

    # Prepare dummy inputs
    dummy_inputs = (
        torch.randint(0, 1000, (1, 64), dtype=torch.long),
        torch.ones((1, 64), dtype=torch.long)
    )

    # Test PyTorch output
    tokenizer = AutoTokenizer.from_pretrained(
        getattr(cfg, "text_tokenizer_name", None), trust_remote_code=True
    )
    tokens = tokenizer.encode("こんにちは", add_special_tokens=False) + [1]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        pt_out = wrapper(input_ids, attention_mask).float().numpy()

    logger.info(f"PyTorch: mean={pt_out.mean():.6f}, std={pt_out.std():.6f}")
    logger.info(f"First 5: {pt_out[0, 0, :5]}")

    # Export to ONNX
    output_path = os.path.join(output_dir, "encoder.onnx")
    logger.info(f"Exporting to {output_path}...")

    torch.onnx.export(
        wrapper,
        dummy_inputs,
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
    logger.info("✅ ONNX export successful!")

    # Verify
    logger.info("\nVerifying ONNX output...")
    import onnxruntime as ort
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])

    logger.info("ONNX Inputs:")
    for inp in session.get_inputs():
        logger.info(f"  {inp.name}: {inp.shape}")

    onnx_out = session.run(None, {
        "input_ids": input_ids.numpy().astype(np.int64),
        "attention_mask": attention_mask.numpy().astype(np.int64),
    })[0]

    logger.info(f"ONNX: mean={onnx_out.mean():.6f}, std={onnx_out.std():.6f}")
    logger.info(f"First 5: {onnx_out[0, 0, :5]}")

    cos_sim = np.dot(pt_out.flatten(), onnx_out.flatten()) / (
        np.linalg.norm(pt_out.flatten()) * np.linalg.norm(onnx_out.flatten()) + 1e-8
    )

    logger.info(f"\nCosine similarity: {cos_sim:.6f}")

    if cos_sim > 0.999:
        logger.info("✅ EXCELLENT! ONNX matches PyTorch perfectly!")
    elif cos_sim > 0.99:
        logger.info("✅ Good match")
    else:
        logger.warning(f"⚠️ Mismatch (cosine sim = {cos_sim:.4f})")

    # Save config
    conf_path = os.path.join(output_dir, "model_args.json")
    conf = cfg.to_dict() if hasattr(cfg, 'to_dict') else {}
    with open(conf_path, 'w') as f:
        json.dump(conf, f, indent=2, default=str)

    del model
    gc.collect()

    logger.info(f"\n{'='*70}")
    logger.info("DONE! Fixed encoder saved to:")
    logger.info(f"  {output_path}")


if __name__ == "__main__":
    main()
