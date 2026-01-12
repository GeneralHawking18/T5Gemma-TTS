#!/bin/bash
python3 tensorrt_llm_implementation/build.py --weights_path trt_weights/weights.npz --output_dir engine_output
