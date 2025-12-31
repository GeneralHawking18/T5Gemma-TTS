#!/usr/bin/env python3
"""
Quantize existing FP32 ONNX models to INT8.
No model loading required - just reads ONNX files and quantizes them.

Usage: python quantize_only.py --input_dir ./onnx_fp32 --output_dir ./onnx_int8
"""
import os
import logging

# Set TMPDIR to HDD to avoid "No space left on device"
_HDD_TMP = "/hdd/chungha/hdd-data/tmp"
os.makedirs(_HDD_TMP, exist_ok=True)
os.environ["TMPDIR"] = _HDD_TMP
os.environ["TEMP"] = _HDD_TMP
os.environ["TMP"] = _HDD_TMP

import tempfile
tempfile.tempdir = _HDD_TMP

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logging.info(f"TMPDIR set to: {_HDD_TMP}")


def quantize_to_int8(input_path: str, output_path: str):
    """Quantize an ONNX model to INT8."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logging.error("onnxruntime not installed. Run: pip install onnxruntime")
        return False

    logging.info(f"Quantizing: {input_path} -> {output_path}")
    try:
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8,
        )
        logging.info(f"✓ Success: {output_path}")
        return True
    except Exception as e:
        logging.error(f"✗ Failed: {e}")
        return False


def main(
    input_dir: str = "./onnx_fp32",
    output_dir: str = "./onnx_int8",
):
    """
    Quantize all FP32 ONNX files in input_dir to INT8.
    """
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .onnx files recursively
    onnx_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".onnx"):
                onnx_files.append(os.path.join(root, f))
    
    if not onnx_files:
        logging.warning(f"No .onnx files found in {input_dir}")
        return
    
    logging.info(f"Found {len(onnx_files)} ONNX file(s):")
    for f in onnx_files:
        logging.info(f"  - {f}")
    
    # Quantize each
    success = 0
    for onnx_path in onnx_files:
        basename = os.path.basename(onnx_path)
        name_no_ext = os.path.splitext(basename)[0]
        out_path = os.path.join(output_dir, f"{name_no_ext}.int8.onnx")
        
        if os.path.exists(out_path):
            logging.info(f"[CACHE] {out_path} already exists. Skipping.")
            success += 1
            continue
        
        if quantize_to_int8(onnx_path, out_path):
            success += 1
    
    logging.info("=" * 50)
    logging.info(f"Done! {success}/{len(onnx_files)} files quantized.")
    logging.info(f"Output: {output_dir}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
