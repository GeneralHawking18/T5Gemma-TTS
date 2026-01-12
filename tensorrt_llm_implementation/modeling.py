import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.functional import (
    gelu,
    concat,
    matmul,
    softmax,
    tanh,
    sin,
    cos,
    split,
    mul,
    add,
    cast,
    arange,
    pow,
    unsqueeze,
    repeat_interleave,
    view,
    shape,
    transpose,
    exp,
    constant,
    cumsum,
    expand,
    mean,
    sqrt,
)
import math
import numpy as np
from tensorrt_llm.layers import Linear, Embedding, RmsNorm
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm import Parameter


def rotate_half(x, half_dim=128):
    x1, x2 = split(x, half_dim, dim=-1)
    neg_x2 = mul(x2, -1.0)
    return concat([neg_x2, x1], dim=-1)


def apply_rope_fixed(x, seq_len_int, head_dim, rope_theta=10000.0):
    dtype = x.dtype
    half_dim = head_dim // 2
    idx = mul(arange(0, half_dim, dtype="float32"), 2.0)
    neg_log_scale = -math.log(rope_theta) / float(head_dim)
    inv_freq = exp(mul(idx, neg_log_scale))
    positions = arange(0, seq_len_int, dtype="float32")
    pos_expanded = unsqueeze(positions, -1)
    inv_freq_expanded = unsqueeze(inv_freq, 0)
    angles = mul(pos_expanded, inv_freq_expanded)
    angles = concat([angles, angles], dim=-1)
    cos_emb = cast(cos(angles), dtype)
    sin_emb = cast(sin(angles), dtype)
    cos_emb = unsqueeze(unsqueeze(cos_emb, 0), 2)
    sin_emb = unsqueeze(unsqueeze(sin_emb, 0), 2)
    return add(mul(x, cos_emb), mul(rotate_half(x, half_dim), sin_emb))


def apply_pm_rope(x, position_ids, head_dim, rope_theta=10000.0):
    dtype = x.dtype
    half_dim = head_dim // 2
    idx = mul(arange(0, half_dim, dtype="float32"), 2.0)
    neg_log_scale = -math.log(rope_theta) / float(head_dim)
    inv_freq = exp(mul(idx, neg_log_scale))
    pos_expanded = unsqueeze(position_ids, -1)
    inv_freq_expanded = unsqueeze(unsqueeze(inv_freq, 0), 0)
    angles = mul(pos_expanded, inv_freq_expanded)
    angles = concat([angles, angles], dim=-1)
    cos_emb = cast(cos(angles), dtype)
    sin_emb = cast(sin(angles), dtype)
    cos_emb = unsqueeze(cos_emb, 2)
    sin_emb = unsqueeze(sin_emb, 2)
    return add(mul(x, cos_emb), mul(rotate_half(x, half_dim), sin_emb))


class T5GemmaAttention(Module):
    def __init__(self, config, layer_idx, is_cross=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_cross = is_cross
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_kv_heads", config.num_attention_heads)
        self.head_dim = config.d_kv
        self.q_inner_dim = self.num_heads * self.head_dim
        self.kv_inner_dim = self.num_kv_heads * self.head_dim
        self.num_groups = self.num_heads // self.num_kv_heads
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.query_pre_attn_scalar = getattr(config, "query_pre_attn_scalar", 256.0)
        self.attn_logit_softcapping = getattr(config, "attn_logit_softcapping", 50.0)

        self.q = Linear(
            self.hidden_size, self.q_inner_dim, bias=False, dtype="bfloat16"
        )
        self.k = Linear(
            self.hidden_size, self.kv_inner_dim, bias=False, dtype="bfloat16"
        )
        self.v = Linear(
            self.hidden_size, self.kv_inner_dim, bias=False, dtype="bfloat16"
        )
        self.o = Linear(
            self.q_inner_dim, self.hidden_size, bias=False, dtype="bfloat16"
        )

    def forward(
        self,
        hidden_states,
        kv_states=None,
        position_ids=None,
        encoder_position_ids=None,
        mask=None,
    ):
        q = self.q(hidden_states)
        if self.is_cross:
            k = self.k(kv_states)
            v = self.v(kv_states)
        else:
            k = self.k(hidden_states)
            v = self.v(hidden_states)

        b = shape(q, 0)
        s = shape(q, 1)
        kv_s = shape(k, 1)
        q = view(q, concat([b, s, self.num_heads, self.head_dim]))
        k = view(k, concat([b, kv_s, self.num_kv_heads, self.head_dim]))
        v = view(v, concat([b, kv_s, self.num_kv_heads, self.head_dim]))

        if not self.is_cross:
            if position_ids is not None:
                q = apply_pm_rope(q, position_ids, self.head_dim, self.rope_theta)
                k = apply_pm_rope(k, position_ids, self.head_dim, self.rope_theta)
        else:
            if position_ids is not None:
                q = apply_pm_rope(q, position_ids, self.head_dim, self.rope_theta)
            if encoder_position_ids is not None:
                k = apply_pm_rope(
                    k, encoder_position_ids, self.head_dim, self.rope_theta
                )

        if self.num_groups > 1:
            k = repeat_interleave(k, self.num_groups, dim=2)
            v = repeat_interleave(v, self.num_groups, dim=2)

        q = transpose(q, 1, 2)
        k = transpose(k, 1, 2)
        v = transpose(v, 1, 2)

        attn_weights = mul(
            matmul(q, transpose(k, 2, 3)), 1.0 / (self.query_pre_attn_scalar**0.5)
        )

        # 4.1. Softcapping
        if self.attn_logit_softcapping is not None:
            attn_weights = cast(attn_weights, "float32")
            attn_weights = mul(
                tanh(mul(attn_weights, 1.0 / self.attn_logit_softcapping)),
                self.attn_logit_softcapping,
            )

        # 4.2. Causal Mask (Self-Attention only)
        if not self.is_cross:
            # Create dynamic causal mask [S, S]
            # Use cumsum on ones to generate 0, 1, ..., S-1 dynamically
            ones = expand(
                cast(constant(np.array([1.0], dtype="float32")), "int32"), concat([s])
            )
            dynamic_range = add(cumsum(ones, 0), -1)

            # row indices: [S, 1]
            row_idx = unsqueeze(dynamic_range, 1)
            # col indices: [1, S]
            col_idx = unsqueeze(dynamic_range, 0)

            # Mask where col > row (future)
            future_mask = col_idx > row_idx  # [S, S] bool
            future_mask = cast(future_mask, "float32")

            # -1e9 is sufficient for bfloat16/float32 softmax to output 0
            # Broadcast to [1, 1, S, S]
            causal_bias = mul(future_mask, -1e9)
            causal_bias = unsqueeze(unsqueeze(causal_bias, 0), 0)

            attn_weights = add(attn_weights, causal_bias)

        if mask is not None:
            # Assume mask is binary [B, KV_S] (1=keep, 0=pad)
            # Convert to additive mask [B, 1, 1, KV_S]: (1 - mask) * -1e4
            mask_f = cast(mask, "float32")
            mask_f = unsqueeze(unsqueeze(mask_f, 1), 1)
            inv_mask = mul(add(mask_f, -1.0), -1.0)  # 1 - mask
            adder = mul(inv_mask, -1e9)
            attn_weights = add(attn_weights, adder)

        attn_weights = cast(softmax(cast(attn_weights, "float32"), dim=-1), "bfloat16")
        output = matmul(attn_weights, v)
        output = view(transpose(output, 1, 2), concat([b, s, self.q_inner_dim]))
        return self.o(output)


def gelu_approximate(x):
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c1 = 0.7978845608  # sqrt(2/pi)
    c2 = 0.044715

    # x^3
    x3 = pow(x, 3.0)
    # x + 0.044715 * x^3
    inner = add(x, mul(x3, c2))
    # sqrt(2/pi) * (...)
    inner = mul(inner, c1)
    # tanh(...)
    t = tanh(inner)
    # 1 + tanh(...)
    res = add(t, 1.0)
    # 0.5 * x * (...)
    return mul(mul(x, 0.5), res)


class GemmaMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.d_ff
        self.gate = Linear(
            self.hidden_size, self.intermediate_size, bias=False, dtype="bfloat16"
        )
        self.up = Linear(
            self.hidden_size, self.intermediate_size, bias=False, dtype="bfloat16"
        )
        self.down = Linear(
            self.intermediate_size, self.hidden_size, bias=False, dtype="bfloat16"
        )

    def forward(self, x):
        return self.down(mul(gelu_approximate(self.gate(x)), self.up(x)))


class T5GemmaBlock(Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.pre_sa_norm = RmsNorm(config.hidden_size, eps=eps, dtype="bfloat16")
        self.post_sa_norm = RmsNorm(config.hidden_size, eps=eps, dtype="bfloat16")
        self.self_attn = T5GemmaAttention(config, layer_idx, is_cross=False)
        self.pre_ca_norm = RmsNorm(config.hidden_size, eps=eps, dtype="bfloat16")
        self.post_ca_norm = RmsNorm(config.hidden_size, eps=eps, dtype="bfloat16")
        self.cross_attn = T5GemmaAttention(config, layer_idx, is_cross=True)
        self.pre_ff_norm = RmsNorm(config.hidden_size, eps=eps, dtype="bfloat16")
        self.post_ff_norm = RmsNorm(config.hidden_size, eps=eps, dtype="bfloat16")
        self.mlp = GemmaMLP(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        position_ids=None,
        encoder_position_ids=None,
        attention_mask=None,
        encoder_attention_mask=None,
    ):
        res = hidden_states
        normed = self.pre_sa_norm(hidden_states)
        if self.layer_idx == 0:
            normed.mark_output("layer_0_norm_1", "bfloat16")
        sa_out = self.self_attn(normed, position_ids=position_ids, mask=attention_mask)
        if self.layer_idx == 0:
            sa_out.mark_output("layer_0_attn_out", "bfloat16")
        hidden_states = add(res, self.post_sa_norm(sa_out))
        if self.layer_idx == 0:
            hidden_states.mark_output("layer_0_post_sa", "bfloat16")

        if encoder_hidden_states is not None:
            res = hidden_states
            normed = self.pre_ca_norm(hidden_states)
            ca_out = self.cross_attn(
                normed,
                kv_states=encoder_hidden_states,
                position_ids=position_ids,
                encoder_position_ids=encoder_position_ids,
                mask=encoder_attention_mask,
            )
            if self.layer_idx == 0:
                ca_out.mark_output("layer_0_cross_out", "bfloat16")
            hidden_states = add(res, self.post_ca_norm(ca_out))

        res = hidden_states
        normed = self.pre_ff_norm(hidden_states)
        if self.layer_idx == 0:
            normed.mark_output("layer_0_norm_ff", "bfloat16")
        ff_out = self.mlp(normed)
        if self.layer_idx == 0:
            ff_out.mark_output("layer_0_mlp_out", "bfloat16")
        return add(res, self.post_ff_norm(ff_out))


class T5GemmaDecoderTRT(Module):
    def __init__(self, config):
        super().__init__()
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, dtype="bfloat16"
        )
        self.layers = ModuleList(
            [T5GemmaBlock(config, i) for i in range(config.num_decoder_layers)]
        )
        self.final_norm = RmsNorm(config.hidden_size, eps=eps, dtype="bfloat16")

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        position_ids=None,
        encoder_position_ids=None,
        encoder_attention_mask=None,
        attention_mask=None,
    ):
        encoder_hidden_states = cast(encoder_hidden_states, "bfloat16")
        x = self.embed_tokens(input_ids)
        x.mark_output("embedding_out", "bfloat16")
        all_hs = []
        for layer in self.layers:
            x = layer(
                x,
                encoder_hidden_states=encoder_hidden_states,
                position_ids=position_ids,
                encoder_position_ids=encoder_position_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
            all_hs.append(x)
        return self.final_norm(x), all_hs


class T5GemmaDecoderWithLMHead(Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = T5GemmaDecoderTRT(config)
        # LM Head: Linear(2304, 2304) -> GELU -> Linear(2304, 65541)
        # Note: PyTorch Linear has bias=True by default.
        # Check if keys exist in weights.npz. usually they do for this architecture.
        self.lm_head_0 = Linear(
            config.hidden_size, config.hidden_size, bias=True, dtype="bfloat16"
        )
        self.lm_head_2 = Linear(
            config.hidden_size, config.vocab_size, bias=True, dtype="bfloat16"
        )

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        position_ids=None,
        encoder_position_ids=None,
        encoder_attention_mask=None,
        attention_mask=None,
    ):
        hidden_states, _ = self.decoder(
            input_ids,
            encoder_hidden_states,
            position_ids,
            encoder_position_ids,
            encoder_attention_mask,
            attention_mask,
        )

        # LM Head forward
        # predict_layer.0
        x = self.lm_head_0(hidden_states)
        # predict_layer.1 (GELU)
        x = gelu(x)
        # predict_layer.2
        logits = self.lm_head_2(x)
        return logits
