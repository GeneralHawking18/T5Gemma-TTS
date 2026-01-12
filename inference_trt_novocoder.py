
import os
import json
import time
import logging
import argparse
import numpy as np
import torch
import tensorrt as trt
try:
    import tensorrt_llm
except ImportError:
    logger.warning("TensorRT-LLM not found. TRT plugins might not be loaded.")
import onnxruntime as ort
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRT-NoVocoder")

# Local imports
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.utils import topk_sampling

# Mock Audio Tokenizer to avoid download
class MockAudioTokenizer:
    def __init__(self, sample_rate=16000, device='cuda'):
        self.sample_rate = sample_rate
        self.device = device
        logger.info("Initialized MockAudioTokenizer (Vocoder skipped)")

    def decode(self, tokens):
        logger.info("MockAudioTokenizer.decode called - skipping audio generation")
        # Return dummy audio
        return torch.zeros((1, 16000), device=self.device)

# TRT Wrapper
class TRTDecoderWrapper:
    """Wrapper for TensorRT-LLM Decoder Engine"""
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        # Register plugins
        trt.init_libnvinfer_plugins(self.logger, "")
        
        logger.info(f"Loading TRT Decoder engine: {engine_path}")
        with open(engine_path, "rb") as f:
            engine_buffer = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_buffer)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

    def run(self, inputs: dict):
        """Execute TRT engine with given inputs (dict of torch tensors)"""
        # Set input shapes for dynamic dimensions
        for name, tensor in inputs.items():
            self.context.set_input_shape(name, tensor.shape)
            self.context.set_tensor_address(name, tensor.data_ptr())

        # Allocate output buffer
        output_name = "output"
        output_shape = self.context.get_tensor_shape(output_name)
        output_tensor = torch.empty(tuple(output_shape), dtype=torch.bfloat16, device='cuda')
        self.context.set_tensor_address(output_name, output_tensor.data_ptr())

        # Run
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return output_tensor

class TRTInference:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.encoder_session = None
        self.trt_decoder = None
        self.predict_layer = None
        self.progress_scale = None
        self.audio_embedding = None
        
        self.load_model()

    def load_model(self):
        # 1. Load Model Args
        weights_path = self.args.weights_path
        args_path = os.path.join(os.path.dirname(weights_path), "model_args.json")
        logger.info(f"Loading model args from {args_path}")
        with open(args_path, "r") as f:
            model_args_dict = json.load(f)
        self.model_args = type("Args", (), model_args_dict)()

        # 2. Tokenizer
        # FORCE local path. If it doesn't exist, this will crash (as intended) instead of downloading.
        tokenizer_path = "/app/tokenizer_local" 
        
        if not os.path.exists(tokenizer_path):
             logger.error(f"Local tokenizer not found at {tokenizer_path}! Please ensure 'tokenizer_local' is mounted.")
             # Fallback only if you REALLY want to allow download, but user asked to stop it.
             # tokenizer_path = getattr(self.model_args, "t5gemma_model_name", "google/t5gemma-2b-2b-ul2")
             raise FileNotFoundError(f"Local tokenizer not found at {tokenizer_path}")

        logger.info(f"Loading tokenizer from LOCAL path: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 3. ONNX Encoder
        logger.info(f"Loading ONNX Encoder: {self.args.onnx_path}")
        self.encoder_session = ort.InferenceSession(self.args.onnx_path, providers=['CPUExecutionProvider'])

        # 4. TRT Decoder
        logger.info(f"Loading TRT Decoder: {self.args.engine_path}")
        self.trt_decoder = TRTDecoderWrapper(self.args.engine_path)

        # 5. Embeddings & Heads (PyTorch)
        logger.info("Loading PyTorch embeddings and heads...")
        from models.t5gemma import T5GemmaVoiceModel
        temp_model = T5GemmaVoiceModel(self.model_args).to(dtype=torch.bfloat16, device='cuda')
        state_dict = torch.load(weights_path, map_location="cpu")
        temp_model.load_state_dict(state_dict, strict=False)
        
        self.predict_layer = temp_model.predict_layer[0].eval()
        self.progress_scale = temp_model.progress_scale
        
        del temp_model
        torch.cuda.empty_cache()

    def _build_position_ids(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        pos = torch.arange(max_len, device='cuda', dtype=torch.float32)[None, :]
        denom = (lengths.clamp(min=2).float() - 1.0)[:, None]
        return (pos / denom * self.progress_scale).masked_fill(pos >= lengths[:, None], 0.0)

    @torch.inference_mode()
    def synthesize(self, text, output_file="output.npy"):
        # 1. Prepare Text
        from inference_tts_utils import normalize_text_with_lang
        from duration_estimator import estimate_duration
        
        text, lang_code = normalize_text_with_lang(text, self.args.lang)
        
        # Estimate Duration
        target_duration = estimate_duration(target_text=text, target_lang=lang_code)
        logger.info(f"Estimated duration: {target_duration:.2f}s")
        
        target_total_tokens = int(target_duration * self.model_args.encodec_sr)
        max_tokens = min(target_total_tokens + int(self.model_args.encodec_sr), 2048)

        # 2. Encode Text (ONNX)
        text_tokens = self.tokenizer.encode(text.strip(), add_special_tokens=False)
        if getattr(self.model_args, "add_eos_to_text", 0): text_tokens.append(self.model_args.add_eos_to_text)
        if getattr(self.model_args, "add_bos_to_text", 0): text_tokens = [self.model_args.add_bos_to_text] + text_tokens
        
        input_ids_enc = np.array([text_tokens], dtype=np.int64)
        enc_out = self.encoder_session.run(None, {"input_ids": input_ids_enc, "attention_mask": np.ones_like(input_ids_enc)})[0]
        memory = torch.from_numpy(enc_out).to(device='cuda', dtype=torch.bfloat16)
        
        enc_len = torch.tensor([input_ids_enc.shape[1]], device='cuda')
        enc_pos_ids = self._build_position_ids(enc_len, input_ids_enc.shape[1])
        enc_mask = torch.ones((1, input_ids_enc.shape[1]), dtype=torch.int32, device='cuda')

        # 3. Decode Loop (TRT)
        generated_tokens = []
        current_tokens = torch.tensor([[self.model_args.empty_token]], device='cuda', dtype=torch.long)
        est_total = target_total_tokens + 1
        
        logger.info(f"Generating tokens (max {max_tokens})...")
        start_time = time.time()
        
        for i in range(max_tokens):
            cur_len = current_tokens.shape[1]
            pos_base = torch.arange(cur_len, device='cuda', dtype=torch.float32).unsqueeze(0)
            dec_pos_ids = pos_base / max(1, est_total - 1) * self.progress_scale
            
            trt_inputs = {
                "input_ids": current_tokens.to(torch.int32),
                "encoder_hidden_states": memory,
                "position_ids": dec_pos_ids,
                "encoder_position_ids": enc_pos_ids,
                "encoder_attention_mask": enc_mask
            }
            
            hidden_states = self.trt_decoder.run(trt_inputs)
            last_hidden = hidden_states[:, -1:, :]
            
            logits = self.predict_layer(last_hidden).squeeze(0).squeeze(0)
            if i == 0: logits[self.model_args.eog] = -1e9
            
            token = topk_sampling(logits, top_k=self.args.top_k, temperature=self.args.temperature)
            token_id = int(token.item())
            
            if token_id == self.model_args.eog or token_id == getattr(self.model_args, "eos", -1):
                break
            
            generated_tokens.append(token_id)
            current_tokens = torch.cat([current_tokens, token.unsqueeze(0)], dim=1)

        inf_time = time.time() - start_time
        logger.info(f"Generated {len(generated_tokens)} tokens in {inf_time:.2f}s")
        
        # Save tokens
        np.save(output_file, np.array(generated_tokens))
        logger.info(f"Tokens saved to {output_file}")
        
        return generated_tokens

def main():
    parser = argparse.ArgumentParser(description="T5Gemma TRT Inference (No Vocoder)")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--onnx_path", type=str, default="onnx_models_fp16_fixed/encoder.onnx")
    parser.add_argument("--engine_path", type=str, default="tensorrt_llm_implementation/engine_output/t5gemma_decoder_new.engine")
    parser.add_argument("--weights_path", type=str, default="weights/decoder_pmrope.bin")
    parser.add_argument("--output", type=str, default="output.npy")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    args = parser.parse_args()
    
    inference = TRTInference(args)
    inference.synthesize(args.text, args.output)

if __name__ == "__main__":
    main()
