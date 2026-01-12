#!/bin/bash
python3 tensorrt_llm_implementation/verify_outputs.py --weights_dir /app/weights --engine_dir tensorrt_llm_implementation/engine_output
