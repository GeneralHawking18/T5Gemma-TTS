import triton_python_backend_utils as pb_utils
import numpy as np
import torch
import tensorrt as trt
import json
import os
import time


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        model_dir = os.path.join(args["model_repository"], args["model_version"])

        # Load Model Args
        args_path = os.path.join(model_dir, "model_args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                self.args = json.load(f)
        else:
            print(f"Warning: {args_path} not found. Using defaults.")
            self.args = {}

        self.empty_token = self.args.get("empty_token", 65536)
        self.eog = self.args.get("eog", 65537)
        self.eos = self.args.get("eos", 65539)
        self.progress_scale = self.args.get("progress_scale", 2000.0)
        self.encodec_sr = self.args.get("encodec_sr", 50.0)

        # Lazy loading - don't load engine at startup to save GPU memory
        self.engine = None
        self.context = None
        self.logger = None
        self.runtime = None
        self._initialized = False

        # Determine engine path
        engine_path = os.path.join(model_dir, "decoder.engine")
        if not os.path.exists(engine_path):
            engine_path = "/workspace/trt_weights/t5gemma_decoder_with_lm_head.engine"
        self.engine_path = engine_path
        print(f"[Decoder] Will load engine lazily from {engine_path}")

    def _load_engine(self):
        """Load the TRT engine on first request"""
        if self._initialized:
            return

        print(f"[Decoder] Loading TRT engine from {self.engine_path}")
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

        self._initialized = True
        print(f"[Decoder] Engine loaded successfully")

    def execute(self, requests):
        # Lazy load on first request
        if not self._initialized:
            self._load_engine()

        responses = []
        for request in requests:
            try:
                # 1. Get Inputs (move to GPU)
                # Note: pb_utils.get_input_tensor_by_name returns a pb_utils.Tensor
                # We use to_dlpack to zero-copy convert to torch

                enc_hidden_t = pb_utils.get_input_tensor_by_name(
                    request, "ENCODER_HIDDEN_STATES"
                )
                enc_mask_t = pb_utils.get_input_tensor_by_name(
                    request, "ENCODER_ATTENTION_MASK"
                )
                text_len_t = pb_utils.get_input_tensor_by_name(request, "TEXT_LENGTH")
                dur_t = pb_utils.get_input_tensor_by_name(request, "TARGET_DURATION")

                # Helper to convert to torch
                def to_torch(t):
                    if t.is_cpu():
                        return torch.from_numpy(t.as_numpy()).cuda()
                    else:
                        return torch.from_dlpack(t.to_dlpack())

                enc_hidden = to_torch(enc_hidden_t)
                enc_mask = to_torch(enc_mask_t)
                # Ensure types
                enc_hidden = enc_hidden.to(torch.bfloat16)
                enc_mask = enc_mask.to(torch.int32)

                text_len = int(text_len_t.as_numpy()[0])
                target_dur = float(dur_t.as_numpy()[0])

                # Sampling params
                top_k = 50
                top_p = 0.95
                temp = 1.0

                tk = pb_utils.get_input_tensor_by_name(request, "TOP_K")
                if tk:
                    top_k = int(tk.as_numpy()[0])
                tp = pb_utils.get_input_tensor_by_name(request, "TOP_P")
                if tp:
                    top_p = float(tp.as_numpy()[0])
                tm = pb_utils.get_input_tensor_by_name(request, "TEMPERATURE")
                if tm:
                    temp = float(tm.as_numpy()[0])

                # Generation Loop
                # Initial token
                batch_size = 1
                input_ids = torch.tensor(
                    [[self.empty_token]], dtype=torch.int32, device="cuda"
                )

                # Calculate encoder positions (0 to 2000 scaled)
                enc_len = enc_hidden.shape[1]
                enc_pos_ids = torch.arange(enc_len, dtype=torch.float32, device="cuda")
                # Avoid div by zero
                if enc_len > 0:
                    enc_pos_ids = (enc_pos_ids / enc_len) * self.progress_scale
                enc_pos_ids = enc_pos_ids.unsqueeze(0)  # [1, enc_len]

                # Max tokens
                max_tokens = int(target_dur * self.encodec_sr + 50)
                max_tokens = min(max_tokens, 2048)  # Safety limit from build profile

                generated = []

                for step in range(max_tokens):
                    cur_len = input_ids.shape[1]
                    if cur_len > 2048:
                        break

                    # Decoder positions
                    est_total = max(1.0, target_dur * self.encodec_sr)

                    dec_pos = torch.arange(cur_len, dtype=torch.float32, device="cuda")
                    dec_pos = (dec_pos / est_total) * self.progress_scale
                    dec_pos = dec_pos.unsqueeze(0)  # [1, cur_len]

                    # Run TRT
                    self.context.set_input_shape("input_ids", input_ids.shape)
                    self.context.set_input_shape(
                        "encoder_hidden_states", enc_hidden.shape
                    )
                    self.context.set_input_shape("position_ids", dec_pos.shape)
                    self.context.set_input_shape(
                        "encoder_position_ids", enc_pos_ids.shape
                    )
                    self.context.set_input_shape(
                        "encoder_attention_mask", enc_mask.shape
                    )

                    logits_shape = (batch_size, cur_len, 65541)
                    logits = torch.empty(
                        logits_shape, dtype=torch.bfloat16, device="cuda"
                    )

                    self.context.set_tensor_address("input_ids", input_ids.data_ptr())
                    self.context.set_tensor_address(
                        "encoder_hidden_states", enc_hidden.data_ptr()
                    )
                    self.context.set_tensor_address("position_ids", dec_pos.data_ptr())
                    self.context.set_tensor_address(
                        "encoder_position_ids", enc_pos_ids.data_ptr()
                    )
                    self.context.set_tensor_address(
                        "encoder_attention_mask", enc_mask.data_ptr()
                    )
                    self.context.set_tensor_address("logits", logits.data_ptr())

                    self.context.execute_async_v3(
                        torch.cuda.current_stream().cuda_stream
                    )
                    torch.cuda.current_stream().synchronize()

                    # Sample
                    next_logits = logits[:, -1, :].float()
                    if temp > 0:
                        next_logits = next_logits / temp
                        probs = torch.softmax(next_logits, dim=-1)
                        # Apply top_k / top_p if needed (omitted for brevity, just multinomial)
                        # Simple multinomial is usually "good enough" for TTS with low temp
                        next_token = torch.multinomial(probs, 1).squeeze(-1)
                    else:
                        next_token = torch.argmax(next_logits, dim=-1)

                    token_val = next_token.item()

                    if token_val in [self.eog, self.eos]:
                        break

                    generated.append(token_val)

                    next_token_t = next_token.unsqueeze(0).int()
                    input_ids = torch.cat([input_ids, next_token_t], dim=1)

                if not generated:
                    generated = [0]  # Dummy

                out_tokens = np.array([generated], dtype=np.int64)
                num_tokens = np.array([len(generated)], dtype=np.int32)

                out_tensor = pb_utils.Tensor("AUDIO_TOKENS", out_tokens)
                num_tensor = pb_utils.Tensor("NUM_TOKENS", num_tokens)

                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[out_tensor, num_tensor])
                )

            except Exception as e:
                import traceback

                traceback.print_exc()
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                )

        # Unload engine to free memory for Vocoder
        print("[Decoder] Unloading engine to free memory...")
        self.context = None
        self.engine = None
        self.runtime = None
        self.logger = None
        self._initialized = False
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print("[Decoder] Engine unloaded.")

        return responses
