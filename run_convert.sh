#!/bin/bash
python3 tensorrt_llm_implementation/convert_weights.py --model_name weights/t5gemma_decoder_only.bin --output_dir trt_weights
