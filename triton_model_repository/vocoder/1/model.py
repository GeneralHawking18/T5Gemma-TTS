import triton_python_backend_utils as pb_utils
import numpy as np
import torch
import tensorrt as trt
import os


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.join(args["model_repository"], args["model_version"])

        # Lazy loading - don't load engine at startup to save GPU memory
        self.engine = None
        self.context = None
        self.logger = None
        self.runtime = None
        self._initialized = False

        # Determine engine path
        engine_path = os.path.join(model_dir, "model.plan")
        if not os.path.exists(engine_path):
            engine_path = "/workspace/trt_weights/vocoder.engine"
        self.engine_path = engine_path
        print(f"[Vocoder] Will load engine lazily from {engine_path}")

    def _load_engine(self):
        """Load the TRT engine on first request"""
        if self._initialized:
            return

        print(f"[Vocoder] Loading TRT engine from {self.engine_path}")
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

        print(f"[Vocoder] Engine has {self.engine.num_io_tensors} I/O tensors")
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            print(f"  [{i}] {name}: shape={shape}, dtype={dtype}, mode={mode}")

        self._initialized = True
        print(f"[Vocoder] Engine loaded successfully")

    def execute(self, requests):
        # Lazy load on first request
        if not self._initialized:
            self._load_engine()

        responses = []
        for request in requests:
            try:
                # Get inputs - audio codes [batch, 1, seq]
                codes_t = pb_utils.get_input_tensor_by_name(request, "codes")
                codes = torch.from_numpy(codes_t.as_numpy()).cuda()

                # Ensure correct dtype
                codes = codes.to(torch.int64)

                batch_size = codes.shape[0]
                num_codebooks = codes.shape[1] if len(codes.shape) > 2 else 1
                seq_len = codes.shape[-1]

                # Audio length estimate: seq_len * hop_size (typically 882 for 44.1kHz @ 50 tokens/sec)
                hop_size = 882
                audio_len = seq_len * hop_size

                # Set input shapes
                self.context.set_input_shape("codes", codes.shape)

                # Allocate output
                output = torch.empty(
                    batch_size, 1, audio_len, dtype=torch.float32, device="cuda"
                )

                # Set tensor addresses
                self.context.set_tensor_address("codes", codes.data_ptr())
                self.context.set_tensor_address("audio", output.data_ptr())

                # Execute
                self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                torch.cuda.current_stream().synchronize()

                # Return output
                out_np = output.cpu().numpy()
                out_tensor = pb_utils.Tensor("audio", out_np)
                responses.append(
                    pb_utils.InferenceResponse(output_tensors=[out_tensor])
                )

            except Exception as e:
                import traceback

                traceback.print_exc()
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                )

        # Unload engine
        print("[Vocoder] Unloading engine to free memory...")
        self.context = None
        self.engine = None
        self.runtime = None
        self.logger = None
        self._initialized = False
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print("[Vocoder] Engine unloaded.")

        return responses
