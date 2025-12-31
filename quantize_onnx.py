#!/usr/bin/env python3
"""
Quantize ONNX models to INT8 using onnxruntime.quantization.
"""

import os
import glob
import logging
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def quantize_model(model_input: str, model_output: str):
    """
    Quantize an ONNX model to INT8 (Dynamic Quantization).
    """
    logging.info(f"Quantizing {model_input} -> {model_output}...")
    try:
        quantize_dynamic(
            model_input=model_input,
            model_output=model_output,
            weight_type=QuantType.QUInt8, # UINT8 is standard for activations, weights usually vary but QUInt8 is common
        )
        logging.info(f"Success: {model_output}")
    except Exception as e:
        logging.error(f"Failed to quantize {model_input}: {e}")

def main(
    input_dir: str = "./onnx_models",
    output_dir: str = "./onnx_models_int8",
    suffix: str = ".int8.onnx"
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all ONNX files
    onnx_files = glob.glob(os.path.join(input_dir, "*.onnx"))
    # Exclude already quantized files just in case
    onnx_files = [f for f in onnx_files if not f.endswith(suffix) and "int8" not in f]
    
    if not onnx_files:
        logging.warning(f"No ONNX files found in {input_dir}")
        return

    logging.info(f"Found {len(onnx_files)} models to quantize.")
    
    for input_path in onnx_files:
        filename = os.path.basename(input_path)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}{suffix}"
        output_path = os.path.join(output_dir, output_filename)
        
        quantize_model(input_path, output_path)
        
        # Copy auxiliary files (like model_args.json) if they exist
        args_json = os.path.join(input_dir, "model_args.json")
        if os.path.exists(args_json):
            import shutil
            out_args = os.path.join(output_dir, "model_args.json")
            if not os.path.exists(out_args):
                shutil.copy(args_json, out_args)
                logging.info("Copied model_args.json")

    logging.info("Quantization process completed.")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
