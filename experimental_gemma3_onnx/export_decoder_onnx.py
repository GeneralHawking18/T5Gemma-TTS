import ast
import os
from typing import Any, Callable, Optional, Sequence, Union
from dataclasses import dataclass

import numpy as np
import onnx_ir as ir
import torch
from onnx_ir.serde import serialize_model
from onnx_ir.tensor_adapters import to_torch_dtype, TorchTensor
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    QuantFormat,
)
import onnx
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig


SUPPORTED_PRECISIONS = ["q4f16", "q4", "fp16", "fp32"]


def get_io_dtype(precision) -> ir.DataType:
    return ir.DataType.FLOAT if precision in {"q4", "fp32"} else ir.DataType.FLOAT16


def get_onnx_dtype(precision: str) -> ir.DataType:
    if precision in ("q4", "q4f16"):
        return ir.DataType.INT4
    return {"fp32": ir.DataType.FLOAT, "fp16": ir.DataType.FLOAT16}[precision]


 @dataclass
class RopeCacheConfig:
    theta: float
    cos_cache_name: str
    sin_cache_name: str
    is_created: bool = False
    rescale_factors: Any = 1
    cache_length: Optional[int] = None
    mscale: float = 1.0


class Gemma3Model:
    def __init__(self, config: PretrainedConfig, *, precision: str = "fp32"):
        if precision not in SUPPORTED_PRECISIONS:
            raise ValueError(f"Unsupported precision: {precision}. Supported: {SUPPORTED_PRECISIONS}")

        # Model configuration
        self.config = config
        self.model_name_or_path = config._name_or_path
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_attn_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        # Data types
        self.io_dtype: ir.DataType = get_io_dtype(precision)
        self.onnx_dtype: ir.DataType = get_onnx_dtype(precision)
        self.use_fp32_layernorm = True

        # ONNX graph construction state
        graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 21, "com.microsoft": 1}, name="main_graph")
        self.model = ir.Model(graph, ir_version=10, producer_name="huggingface")
        self.values: dict[str, ir.Value] = {}
        self.node_names = set()
        self.embedding_weight_name = None  # For tied embeddings

        # RoPE (Rotary Positional Embedding) configuration
        self.rope_attrs = {
            "rotary_embedding_dim": int(self.head_dim * config.partial_rotary_factor) if getattr(config, "partial_rotary_factor", 1.0) != 1.0 else 0,
            "position_scale": getattr(config, "rope_position_scale", 1),
        }
        self.is_local_layer: Callable[[int], bool] = lambda layer_id: config.layer_types[layer_id] == "sliding_attention"
        self.get_rope_config_key: Callable[[int], str] = lambda layer_id: "local" if self.is_local_layer(layer_id) else "global"
        self.rope_cache_configs = {
            "global": RopeCacheConfig(
                theta=config.rope_theta,
                cos_cache_name="cos_cache_global",
                sin_cache_name="sin_cache_global",
                cache_length=config.max_position_embeddings,
            ),
            "local": RopeCacheConfig(
                theta=config.rope_local_base_freq,
                cos_cache_name="cos_cache_local",
                sin_cache_name="sin_cache_local",
                cache_length=config.max_position_embeddings,
            ),
        }

    def save_model(self, out_path: str):
        model = self.to_int4() if self.onnx_dtype == ir.DataType.INT4 else self.model

        # Ensure the graph is topologically sorted before saving
        model.graph.sort()
        proto = serialize_model(model)

        # Save with external data to handle large models
        data_location = os.path.basename(out_path) + "_data"
        onnx.save_model(proto, out_path, save_as_external_data=True, all_tensors_to_one_file=True, location=data_location)

    def to_int4(self) -> ir.Model:
        quantizer = MatMulNBitsQuantizer(
            model=ir.to_proto(self.model),
            block_size=32,
            is_symmetric=True,
            accuracy_level=4,
            nodes_to_exclude=[],
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=("MatMul",),
        )
        quantizer.process()
        return ir.from_proto(quantizer.model.model)

    def _make_value(self, name: str, dtype: Optional[ir.DataType] = None, shape: Optional[Sequence[Union[int, str]]] = None) -> ir.Value:
        if not name:
            return ir.Value(name="")
        value = self.values.setdefault(name, ir.Value(name=name))
        if dtype is not None:
            value.dtype = ir.DataType(dtype)
        if shape is not None:
            value.shape = ir.Shape(shape)
        return value

    def _make_node(self, op_type: str, inputs: Sequence[str], outputs: Sequence[str], name: str, domain: str = "", **kwargs):
        if name in self.node_names:
            return

        for input_name in inputs:
            if input_name.startswith("/model/constants") and input_name not in self.node_names:
                self._make_constant(input_name)

        input_values = [self._make_value(name) for name in inputs]
        output_values = [self._make_value(name) for name in outputs]
        node = ir.node(op_type, inputs=input_values, attributes=kwargs, domain=domain, outputs=output_values, name=name)
        self.model.graph.append(node)
        self.node_names.add(name)

    def _make_initializer(self, tensor: Union[torch.Tensor, np.ndarray], name: str, to: Optional[ir.DataType] = None):
        if to:
            # Lazily cast tensor to the target data type
            def tensor_func():
                casted_tensor = tensor.to(to_torch_dtype(to))
                return TorchTensor(casted_tensor, name=name)
            ir_tensor = ir.LazyTensor(tensor_func, dtype=to, shape=ir.Shape(tensor.shape), name=name)
        else:
            ir_tensor = ir.tensor(tensor, name=name)

        value = self._make_value(name, ir_tensor.dtype, ir_tensor.shape)
        value.const_value = ir_tensor
        self.model.graph.register_initializer(value)

    # ------------------------------------------------------------------------------------
    # ONNX Operator Helper Methods
    # ------------------------------------------------------------------------------------

    def _make_constant(self, name: str):
        path = name.split("/")
        onnx_dtype = ir.DataType[path[-2]]
        num = ast.literal_eval(path[-1])
        tensor = ir.tensor(num, dtype=onnx_dtype, name=name)
        node_name = name.replace("constants", "constant_nodes")
        self._make_node("Constant", [], [name], name=node_name, value=tensor)
        self._make_value(name, onnx_dtype, shape=tensor.shape)

    def _make_op(self, op_type: str, name: str, inputs: list[str], dtype: ir.DataType, shape: list, **kwargs) -> str:
        output = f"{name}/output_0"
        self._make_node(op_type, inputs, [output], name=name, **kwargs)
        self._make_value(output, dtype, shape)
        return output

    def _make_cast(self, root_input: str, to_dtype: ir.DataType, basename: str, shape: Optional[list] = None) -> str:
        name = f"{basename}/Cast"
        output = f"{name}/output_0"
        self._make_node("Cast", [root_input], [output], name=name, to=to_dtype)
        if shape is None:
            shape = self.values[root_input].shape
        self._make_value(output, to_dtype, shape=shape)
        return output

    def _make_matmul(self, root_input: str, weight: torch.Tensor, basename: str, output_shape_dims: list) -> str:
        output = f"{basename}/output_0"
        weight_name = basename[1:].replace("/", ".") + ".weight"

        if self.onnx_dtype == ir.DataType.INT4:
            # For INT4, quantizer will handle it, for now create FP node
            self._make_initializer(weight.T, weight_name, to=self.io_dtype)
            self._make_node("MatMul", [root_input, weight_name], [output], name=basename)
        else:
            # For FP16/FP32
            self._make_initializer(weight.T, weight_name, to=self.io_dtype)
            self._make_node("MatMul", [root_input, weight_name], [output], name=basename)

        self._make_value(output, self.io_dtype, shape=['batch_size', 'sequence_length', output_shape_dims[-1]])
        return output

    # ------------------------------------------------------------------------------------
    # Model Component Building Methods
    # ------------------------------------------------------------------------------------

    def _build_inputs_and_outputs(self):
        # Define shapes and types
        input_shapes = {
            "input_ids": ["batch_size", "sequence_length"],
            "attention_mask": ["batch_size", "total_sequence_length"],
            "position_ids": ["batch_size", "sequence_length"],
            "past_key_values": ["batch_size", self.num_kv_heads, "past_sequence_length", self.head_dim],
        }
        output_shapes = {
            "logits": ["batch_size", "sequence_length", self.vocab_size],
            "present": ["batch_size", self.num_kv_heads, "total_sequence_length", self.head_dim],
        }

        # Create main inputs
        self.model.graph.inputs.extend([
            self._make_value("input_ids", ir.DataType.INT64, input_shapes["input_ids"]),
            self._make_value("attention_mask", ir.DataType.INT64, input_shapes["attention_mask"]),
            self._make_value("position_ids", ir.DataType.INT64, input_shapes["position_ids"]),
        ])

        # Create main output
        self.model.graph.outputs.append(self._make_value("logits", ir.DataType.FLOAT, output_shapes["logits"]))

        # Create KV cache inputs and outputs for each layer
        for i in range(self.num_layers):
            self.model.graph.inputs.extend([
                self._make_value(f"past_key_values.{i}.key", self.io_dtype, input_shapes["past_key_values"]),
                self._make_value(f"past_key_values.{i}.value", self.io_dtype, input_shapes["past_key_values"]),
            ])
            self.model.graph.outputs.extend([
                self._make_value(f"present.{i}.key", self.io_dtype, output_shapes["present"]),
                self._make_value(f"present.{i}.value", self.io_dtype, output_shapes["present"]),
            ])

    def _build_embedding(self, embedding_weight: torch.Tensor) -> str:
        weight_name = "model.embed_tokens.weight"
        self._make_initializer(embedding_weight, weight_name, to=self.io_dtype)
        self.embedding_weight_name = weight_name

        basename = "/model/embed_tokens"
        gather_out = self._make_op("Gather", f"{basename}/Gather", [weight_name, 'input_ids'], self.io_dtype, ['batch_size', 'sequence_length', self.hidden_size])

        # Scale the embeddings
        scale_const = f"/model/constants/{self.io_dtype.name}/{np.sqrt(self.hidden_size)}"
        return self._make_op("Mul", f"{basename}/Mul", [gather_out, scale_const], self.io_dtype, ['batch_size', 'sequence_length', self.hidden_size])

    def _build_layernorm(self, root_input: str, weight: torch.Tensor, basename: str, is_final_norm: bool = False, shape_override: Optional[list] = None) -> str:
        norm_dtype = ir.DataType.FLOAT if self.use_fp32_layernorm else self.io_dtype
        output_shape = shape_override if shape_override is not None else self.values[root_input].shape

        # Add offset to weight and create initializer
        weight_name = basename.replace("/", ".")[1:] + ".weight"
        # Always use float32 for the addition to avoid precision loss with bfloat16
        offset_weight = weight.to(torch.float32) + 1.0
        self._make_initializer(offset_weight, weight_name, to=norm_dtype)

        # Cast input if necessary
        input_to_norm = root_input
        if self.use_fp32_layernorm and self.values[root_input].dtype != norm_dtype:
            if not is_final_norm:
                input_to_norm = self._make_cast(root_input, norm_dtype, basename)

        # Create the LayerNorm node
        norm_output = self._make_op(
            "SimplifiedLayerNormalization", basename, [input_to_norm, weight_name],
            norm_dtype, output_shape,
            epsilon=self.config.rms_norm_eps, axis=-1, stash_type=1
        )

        # Cast output back if necessary
        if self.use_fp32_layernorm and norm_dtype != self.io_dtype:
            return self._make_cast(norm_output, self.io_dtype, f"{basename}/output", shape=output_shape)

        return norm_output

    def _create_rope_cache_tensors(self, config: RopeCacheConfig) -> tuple[torch.Tensor, torch.Tensor]:
        dim = self.rope_attrs["rotary_embedding_dim"] or self.head_dim
        inv_freq = 1.0 / (config.rescale_factors * (config.theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)))
        t = (torch.arange(config.cache_length, dtype=torch.float32) * self.rope_attrs["position_scale"])
        freqs = torch.outer(t, inv_freq)
        cos_cache = (freqs.cos() * config.mscale).to(to_torch_dtype(self.io_dtype))
        sin_cache = (freqs.sin() * config.mscale).to(to_torch_dtype(self.io_dtype))
        return cos_cache, sin_cache

    def _ensure_rope_caches_created(self, layer_id: int) -> tuple[str, str]:
        config = self.rope_cache_configs[self.get_rope_config_key(layer_id)]
        if not config.is_created:
            cos_tensor, sin_tensor = self._create_rope_cache_tensors(config)
            self._make_initializer(cos_tensor, config.cos_cache_name)
            self._make_initializer(sin_tensor, config.sin_cache_name)
            config.is_created = True
        return config.cos_cache_name, config.sin_cache_name

    def _build_rotary_embedding(self, root_input: str, position_ids: str, layer_id: int, num_heads: int, basename: str) -> str:
        cos_cache, sin_cache = self._ensure_rope_caches_created(layer_id)
        inputs = [root_input, position_ids, cos_cache, sin_cache]
        return self._make_op(
            "RotaryEmbedding", basename, inputs, self.io_dtype,
            ['batch_size', 'sequence_length', num_heads * self.head_dim],
            domain="com.microsoft",
            interleaved=0,
            num_heads=num_heads if self.rope_attrs["rotary_embedding_dim"] else 0,
            rotary_embedding_dim=self.rope_attrs["rotary_embedding_dim"],
        )

    def _build_qk_norm(self, q_proj: str, k_proj: str, q_norm_w: torch.Tensor, k_norm_w: torch.Tensor, layer_id: int) -> tuple[str, str]:
        def _norm_path(proj_in: str, weight: torch.Tensor, basename: str, num_heads: int):
            # Reshape from BxSxD to Bx(S*N)xH
            shape_const = f"/model/constants/INT64/[0, -1, {self.head_dim}]"
            reshaped1 = self._make_op("Reshape", f"{basename}/Reshape1", [proj_in, shape_const], self.io_dtype, ['batch_size', -1, self.head_dim])

            # LayerNorm
            normed = self._build_layernorm(reshaped1, weight, f"{basename}/LayerNorm", shape_override=['batch_size', -1, self.head_dim])

            # Reshape back to BxSxD
            shape_const2 = f"/model/constants/INT64/[0, -1, {num_heads * self.head_dim}]"
            return self._make_op("Reshape", f"{basename}/Reshape2", [normed, shape_const2], self.io_dtype, ['batch_size', 'sequence_length', num_heads * self.head_dim])

        q_norm_out = _norm_path(q_proj, q_norm_w, f"/model/layers.{layer_id}/attn/q_norm", self.num_attn_heads)
        k_norm_out = _norm_path(k_proj, k_norm_w, f"/model/layers.{layer_id}/attn/k_norm", self.num_kv_heads)
        return q_norm_out, k_norm_out

    def _build_attention(self, root_input: str, layer_id: int, layer_torch, seqlens_k: str, total_seq_len: str, pos_ids_reformatted: str) -> str:
        basename = f"/model/layers.{layer_id}/attn"
        attn = layer_torch.self_attn

        # Q, K, V projections
        q_proj = self._make_matmul(root_input, attn.q_proj.weight, f"{basename}/q_proj/MatMul", [self.num_attn_heads * self.head_dim])
        k_proj = self._make_matmul(root_input, attn.k_proj.weight, f"{basename}/k_proj/MatMul", [self.num_kv_heads * self.head_dim])
        v_proj = self._make_matmul(root_input, attn.v_proj.weight, f"{basename}/v_proj/MatMul", [self.num_kv_heads * self.head_dim])

        # QK Normalization
        q_proj, k_proj = self._build_qk_norm(q_proj, k_proj, attn.q_norm.weight, attn.k_norm.weight, layer_id)

        # Rotary Embeddings
        q_rotary = self._build_rotary_embedding(q_proj, pos_ids_reformatted, layer_id, self.num_attn_heads, f"{basename}/q_rotary")
        k_rotary = self._build_rotary_embedding(k_proj, pos_ids_reformatted, layer_id, self.num_kv_heads, f"{basename}/k_rotary")

        # GroupQueryAttention
        past_k, past_v = f"past_key_values.{layer_id}.key", f"past_key_values.{layer_id}.value"
        present_k, present_v = f"present.{layer_id}.key", f"present.{layer_id}.value"

        attn_output = f"{basename}/GroupQueryAttention/output_0"
        self._make_node(
            "GroupQueryAttention",
            inputs=[q_rotary, k_rotary, v_proj, past_k, past_v, seqlens_k, total_seq_len, "", "", "", ""],
            outputs=[attn_output, present_k, present_v],
            name=f"{basename}/GroupQueryAttention",
            domain="com.microsoft",
            num_heads=self.num_attn_heads,
            kv_num_heads=self.num_kv_heads,
            scale=self.config.query_pre_attn_scalar ** -0.5,
            local_window_size=self.config.sliding_window if self.is_local_layer(layer_id) else -1,
            softcap=0.0,
            do_rotary=False,
            rotary_interleaved=0
        )
        attn_output_dim = self.num_attn_heads * self.head_dim
        self._make_value(attn_output, self.io_dtype, ['batch_size', 'sequence_length', attn_output_dim])

        # Output projection
        return self._make_matmul(attn_output, attn.o_proj.weight, f"{basename}/o_proj/MatMul", [self.hidden_size])

    def _build_mlp(self, root_input: str, layer_id: int, mlp_torch) -> str:
        basename = f"/model/layers.{layer_id}/mlp"

        # Gate and Up projections
        gate_proj = self._make_matmul(root_input, mlp_torch.gate_proj.weight, f"{basename}/gate_proj/MatMul", [self.config.intermediate_size])
        up_proj = self._make_matmul(root_input, mlp_torch.up_proj.weight, f"{basename}/up_proj/MatMul", [self.config.intermediate_size])

        # Activation function (FastGelu)
        activated_gate = self._make_op("Gelu", f"{basename}/act_fn/FastGelu", [gate_proj], self.io_dtype, ['batch_size', 'sequence_length', self.config.intermediate_size], approximate="tanh")

        # Element-wise multiplication
        mul_out = self._make_op("Mul", f"{basename}/Mul", [activated_gate, up_proj], self.io_dtype, ['batch_size', 'sequence_length', self.config.intermediate_size])

        # Down projection
        return self._make_matmul(mul_out, mlp_torch.down_proj.weight, f"{basename}/down_proj/MatMul", [self.hidden_size])

    def _build_decoder_layer(self, hidden_states: str, layer_id: int, layer_torch, seqlens_k: str, total_seq_len: str, pos_ids_reformatted: str) -> str:
        residual = hidden_states
        basename = f"/model/layers.{layer_id}"

        # Attention block with pre-normalization and residual connection
        norm1_out = self._build_layernorm(hidden_states, layer_torch.input_layernorm.weight, f"{basename}/input_layernorm")
        attn_out = self._build_attention(norm1_out, layer_id, layer_torch, seqlens_k, total_seq_len, pos_ids_reformatted)
        norm2_out = self._build_layernorm(attn_out, layer_torch.post_attention_layernorm.weight, f"{basename}/post_attention_layernorm")
        residual2 = self._make_op("Add", f"{basename}/Add_1", [residual, norm2_out], self.io_dtype, self.values[residual].shape)

        # MLP block with pre-normalization and residual connection
        norm3_out = self._build_layernorm(residual2, layer_torch.pre_feedforward_layernorm.weight, f"{basename}/pre_feedforward_layernorm")
        mlp_out = self._build_mlp(norm3_out, layer_id, layer_torch.mlp)
        norm4_out = self._build_layernorm(mlp_out, layer_torch.post_feedforward_layernorm.weight, f"{basename}/post_feedforward_layernorm")

        return self._make_op("Add", f"{basename}/Add_2", [residual2, norm4_out], self.io_dtype, self.values[residual2].shape)

    def _build_lm_head(self, hidden_states: str):
        if not self.tie_word_embeddings:
            raise NotImplementedError("Only tied embeddings are currently supported.")

        # Transpose the embedding weights: [vocab_size, hidden_size] -> [hidden_size, vocab_size]
        transposed_weight = self._make_op("Transpose", "/lm_head/Transpose", [self.embedding_weight_name], self.io_dtype, [self.hidden_size, self.vocab_size], perm=[1, 0])
        logits_dtype = ir.DataType.FLOAT  # The final logits output is always float32
        logits_shape = ['batch_size', 'sequence_length', self.vocab_size]
        cast_needed = self.io_dtype != logits_dtype

        # Determine the output name for the MatMul operation.
        matmul_output_name = "logits" if not cast_needed else "/lm_head/MatMul/output_0"
        matmul_output_dtype = self.io_dtype

        # Create the MatMul node for the language model head.
        self._make_node("MatMul", [hidden_states, transposed_weight], [matmul_output_name], name="/lm_head/MatMul")
        self._make_value(matmul_output_name, matmul_output_dtype, logits_shape)

        if cast_needed:
            # Create a cast node to convert the MatMul output to float32 for the final logits.
            self._make_node("Cast", [matmul_output_name], ["logits"], name="/lm_head/CastToFloat", to=logits_dtype)
            self._make_value("logits", logits_dtype, logits_shape)

    def _build_preprocessing(self) -> tuple[str, str, str]:
        basename = "/model/preprocessing"

        # Attention mask reformatting for GroupQueryAttention
        reduce_sum_out = self._make_op("ReduceSum", f"{basename}/attn_mask/ReduceSum", ["attention_mask", "/model/constants/INT64/[1]"], ir.DataType.INT64, ["batch_size", 1])
        sub_out = self._make_op("Sub", f"{basename}/attn_mask/Sub", [reduce_sum_out, "/model/constants/INT64/[1]"], ir.DataType.INT64, ["batch_size", 1])
        seqlens_k = self._make_cast(sub_out, ir.DataType.INT32, f"{basename}/attn_mask/seqlens_k_cast")

        shape_out = self._make_op("Shape", f"{basename}/attn_mask/Shape", ["attention_mask"], ir.DataType.INT64, [2])
        gather_out = self._make_op("Gather", f"{basename}/attn_mask/Gather", [shape_out, "/model/constants/INT64/1"], ir.DataType.INT64, [], axis=0)
        total_seq_len = self._make_cast(gather_out, ir.DataType.INT32, f"{basename}/attn_mask/total_seq_len_cast")

        # Position IDs reformatting
        shape_ids_out = self._make_op("Shape", f"{basename}/pos_ids/Shape", ["input_ids"], ir.DataType.INT64, [2])
        gather_ids_out = self._make_op("Gather", f"{basename}/pos_ids/Gather", [shape_ids_out, "/model/constants/INT64/1"], ir.DataType.INT64, [], axis=0)
        unsqueeze_out = self._make_op("Unsqueeze", f"{basename}/pos_ids/Unsqueeze", [gather_ids_out, "/model/constants/INT64/[0]"], ir.DataType.INT64, [1])
        concat_out = self._make_op("Concat", f"{basename}/pos_ids/Concat", ["/model/constants/INT64/[-1]", unsqueeze_out], ir.DataType.INT64, [2], axis=0)
        pos_ids_reformatted = self._make_op("Reshape", f"{basename}/pos_ids/Reshape", ["position_ids", concat_out], ir.DataType.INT64, None)

        return seqlens_k, total_seq_len, pos_ids_reformatted

    def build_model(self):
        # 1. Setup graph inputs, outputs, and preprocessing steps
        self._build_inputs_and_outputs()
        seqlens_k, total_seq_len, pos_ids_reformatted = self._build_preprocessing()

        # 2. Load the source PyTorch model
        print("Loading PyTorch model...")
        torch_model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, dtype="auto", device_map="cpu")

        # 3. Build embedding layer
        print("Building embedding layer...")
        hidden_states = self._build_embedding(torch_model.model.embed_tokens.weight)

        # 4. Build all decoder layers sequentially
        layers = torch_model.model.layers
        for i, layer in enumerate(layers):
            print(f"Building layer {i}/{self.num_layers - 1}...")
            hidden_states = self._build_decoder_layer(hidden_states, i, layer, seqlens_k, total_seq_len, pos_ids_reformatted)

        # 5. Build final normalization layer
        print("Building final normalization layer...")
        final_norm_out = self._build_layernorm(hidden_states, torch_model.model.norm.weight, "/model/final_norm", is_final_norm=True)

        # 6. Build language model head
        print("Building LM head...")
        self._build_lm_head(final_norm_out)

        print("ONNX model construction complete.")
        del torch_model


 @torch.no_grad
def create_model(
    config: PretrainedConfig,
    precision: str,
):
    options = dict(precision=precision)
    architecture = config.architectures[0]

    assert architecture == "Gemma3ForCausalLM", f"Architecture {architecture} is not supported"
    onnx_model = Gemma3Model(config, **options)
    return onnx_model

if __name__ == "__main__":
    import argparse
    import json
    from transformers import GenerationConfig, AutoTokenizer

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        help="Model name in Hugging Face. Do not use if providing an input path to a Hugging Face directory in -i/--input.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Path to folder to store generated files",
    )
    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        choices=SUPPORTED_PRECISIONS,
        nargs='+',
        help="Precision(s) of model. You can specify multiple precisions separated by space.",
    )

    args = parser.parse_args()

    model_name = args.model_name
    output_dir = args.output
    precision = args.precision

    # Support list of precisions
    precisions = [precision] if isinstance(precision, str) else precision
    if len(precisions) == 0:
        raise ValueError("At least one precision must be specified.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving config and processing files in {output_dir}")
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(output_dir)

    generation_config = GenerationConfig.from_pretrained(config._name_or_path)
    generation_config.save_pretrained(output_dir)

    processor = AutoTokenizer.from_pretrained(config._name_or_path)
    processor.save_pretrained(output_dir)

    # Post-process tokenizer config
    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as fp:
            tokenizer_config = json.load(fp)
        if hasattr(processor, "chat_template") and processor.chat_template is not None:
            tokenizer_config["chat_template"] = processor.chat_template
        with open(tokenizer_config_path, "w") as fp:
            json.dump(tokenizer_config, fp, indent=2)

    # Minify tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r") as fp:
            tokenizer_json = json.load(fp)
        with open(tokenizer_path, "w") as fp:
            json.dump(tokenizer_json, fp)

    for precision in precisions:
        precision = precision.lower()
        if precision == "fp32":
            filename = "model.onnx"
        else:
            filename = f"model_{precision}.onnx"

        onnx_model = create_model(config, precision)

        # Build ONNX model
        onnx_model.build_model()

        # Save ONNX model
        model_output_path = os.path.join(output_dir, "onnx")
        os.makedirs(model_output_path, exist_ok=True)
        onnx_model.save_model(os.path.join(model_output_path, filename))

    # Post-process config files
    with open(os.path.join(output_dir, "config.json"), "r") as fp:
        config_data = json.load(fp)
    config_data["transformers.js_config"] = {
        "dtype": "fp32",
        "use_external_data_format": True,
    }
    config_data["transformers.js_config"]["kv_cache_dtype"] = {
        "q4f16": "float16",
        "fp16": "float16"
    }
    with open(os.path.join(output_dir, "config.json"), "w") as fp:
        json.dump(config_data, fp, indent=2)
