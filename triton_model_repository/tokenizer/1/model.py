import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import AutoTokenizer
import os
import json
import unicodedata


class TritonPythonModel:
    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        tokenizer_path = os.path.join(
            args["model_repository"], args["model_version"], "tokenizer_files"
        )
        print(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # SPP (Seconds per character/phoneme) estimates
        self.spp = {"ja": 0.12, "en": 0.08, "zh": 0.15, "default": 0.1}

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # Get Input Text
                in_text = pb_utils.get_input_tensor_by_name(request, "TEXT")
                text_batch = in_text.as_numpy()  # [Batch, 1]

                texts = []
                for t in text_batch.flatten():
                    if isinstance(t, bytes):
                        texts.append(t.decode("utf-8"))
                    else:
                        texts.append(str(t))

                # Get Language (Optional)
                in_lang = pb_utils.get_input_tensor_by_name(request, "LANGUAGE")
                lang = "ja"
                if in_lang is not None:
                    # Use first element for simplicity or map?
                    # Ideally we handle per-sample.
                    l_obj = in_lang.as_numpy().flatten()[0]
                    if isinstance(l_obj, bytes):
                        l_str = l_obj.decode("utf-8")
                    else:
                        l_str = str(l_obj)
                    if l_str:
                        lang = l_str

                # Basic Normalization
                texts = [unicodedata.normalize("NFKC", t) for t in texts]

                # Tokenize
                inputs = self.tokenizer(
                    texts, return_tensors="np", padding=True, truncation=False
                )
                input_ids = inputs["input_ids"].astype(np.int64)
                attention_mask = inputs["attention_mask"].astype(np.int64)

                # Calculate Duration
                # Simple heuristic: len(text) * spp
                spp_val = self.spp.get(lang, self.spp["default"])

                durations = []
                lengths = []

                for i, t in enumerate(texts):
                    d = float(len(t) * spp_val)
                    d = max(d, 1.0)
                    durations.append(d)

                    # Length excluding padding?
                    # input_ids[i] might contain padding if batch > 1
                    # We can use attention mask sum
                    l = np.sum(attention_mask[i])
                    lengths.append(l)

                # Create Output Tensors
                # input_ids is [Batch, Seq]

                out_input_ids = pb_utils.Tensor("INPUT_IDS", input_ids)
                out_attn_mask = pb_utils.Tensor("ATTENTION_MASK", attention_mask)

                # [Batch, 1]
                text_len = np.array(lengths, dtype=np.int32).reshape(-1, 1)
                out_len = pb_utils.Tensor("TEXT_LENGTH", text_len)

                # [Batch, 1]
                target_dur = np.array(durations, dtype=np.float32).reshape(-1, 1)
                out_dur = pb_utils.Tensor("TARGET_DURATION", target_dur)

                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[out_input_ids, out_attn_mask, out_len, out_dur]
                    )
                )
            except Exception as e:
                print(f"Error in tokenizer: {e}")
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                )

        return responses
