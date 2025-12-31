import torch
import torch.nn as nn
from typing import Tuple, Optional, List

# =============================================================================
# Cache Adapter for ONNX Export
# =============================================================================

class TupleCache:
    """
    Adapter class that wraps a flat tuple of K/V tensors to provide 
    the full Cache interface expected by T5Gemma decoder.
    
    Input tuple format: (k0, v0, k1, v1, ..., kN, vN)
    Each k, v has shape: [batch, num_heads, seq_len, head_dim]
    """
    
    def __init__(self, past_kv_tuple: Tuple[torch.Tensor, ...]):
        self._past = list(past_kv_tuple)
        self._num_layers = len(past_kv_tuple) // 2
        
        # Build key_cache and value_cache lists (DynamicCache interface)
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        for i in range(self._num_layers):
            self.key_cache.append(past_kv_tuple[i * 2])
            self.value_cache.append(past_kv_tuple[i * 2 + 1])
        
        # T5Gemma cross-attention tracking
        self.is_updated: dict = {i: True for i in range(self._num_layers)}
        
        # Additional cache attributes
        self._seen_tokens = self.get_seq_length() if self._num_layers > 0 else 0
        
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the sequence length of cached keys."""
        if len(self.key_cache) > layer_idx:
            return self.key_cache[layer_idx].shape[2]
        return 0
    
    def __len__(self) -> int:
        """Return number of layers."""
        return self._num_layers
    
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (key, value) pair for a layer."""
        return (self.key_cache[layer_idx], self.value_cache[layer_idx])
    
    def __iter__(self):
        """Iterate over (key, value) pairs."""
        for i in range(self._num_layers):
            yield self[i]
    
    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """For compatibility with some cache implementations."""
        return self.get_seq_length(layer_idx)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache for a layer - concatenate new states with past."""
        if layer_idx < len(self.key_cache):
            new_key = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            new_value = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
            self.key_cache[layer_idx] = new_key
            self.value_cache[layer_idx] = new_value
        else:
            # Append new layer
            new_key = key_states
            new_value = value_states
            self.key_cache.append(new_key)
            self.value_cache.append(new_value)
            self._num_layers = len(self.key_cache)
            
        # Mark as updated for cross-attention tracking
        self.is_updated[layer_idx] = True
        self._seen_tokens = self.get_seq_length()
        
        return new_key, new_value
    
    def to_tuple(self) -> Tuple[torch.Tensor, ...]:
        """Convert back to flat tuple."""
        result = []
        for i in range(len(self.key_cache)):
            result.append(self.key_cache[i])
            result.append(self.value_cache[i])
        return tuple(result)
    
    @property
    def self_attention_cache(self):
        """T5Gemma expects this property for decoder forward pass."""
        return self
    
    @property
    def cross_attention_cache(self):
        """T5Gemma expects this property for cross-attention in decoder."""
        return None
    
    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0):
        """Return (kv_length, kv_offset) for masking."""
        kv_length = self.get_seq_length(layer_idx) + cache_position.shape[0]
        kv_offset = 0
        return kv_length, kv_offset
    
    def get_max_cache_shape(self):
        """Return max cache shape (for compatibility)."""
        return None
    
    @property
    def is_sliding(self) -> List[bool]:
        """Return list of sliding window flags per layer - T5Gemma iterates over this."""
        return [False] * self._num_layers
    
    @property
    def seen_tokens(self) -> int:
        """Return total tokens seen so far."""
        return self._seen_tokens


# =============================================================================
# ONNX Wrapper Modules
# =============================================================================

class EncoderWrapper(nn.Module):
    """Wrapper for the encoder part of T5Gemma for ONNX export."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.encoder = model.encoder_module
        self.text_input_type = model.text_input_type
        self.text_embedding = model.text_embedding
        self.text_dropout = model.text_dropout
        self.progress_scale = model.progress_scale
        
    def _build_position_ids(
        self, 
        x_lens: torch.Tensor, 
        max_len: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Build PM-RoPE position IDs."""
        lengths = x_lens.to(device=device)
        pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
        denom = (lengths.clamp(min=2).to(torch.float32) - 1.0)[:, None]
        position_ids = pos / denom * self.progress_scale
        mask = pos < lengths[:, None]
        return position_ids.masked_fill(~mask, 0.0)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode text to hidden states.
        
        Args:
            input_ids: [batch, seq_len] - tokenized text
            attention_mask: [batch, seq_len] - 1 for valid tokens, 0 for padding
            
        Returns:
            encoder_hidden_states: [batch, seq_len, hidden_size]
        """
        x_lens = attention_mask.sum(dim=1)
        use_pm_rope = getattr(self.model.args, "use_pm_rope", 1)
        
        position_ids = None
        if use_pm_rope:
            position_ids = self._build_position_ids(
                x_lens, input_ids.shape[1], input_ids.device
            )
            
        if self.text_input_type == "text":
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            x_embeds = self.text_dropout(self.text_embedding(input_ids))
            encoder_outputs = self.encoder(
                inputs_embeds=x_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            
        return encoder_outputs.last_hidden_state


class DecoderInitWrapper(nn.Module):
    """Wrapper for initial decoder pass (process prompt and get initial state)."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.decoder = model.decoder_module
        self.audio_embedding = model.audio_embedding[0]
        self.audio_dropout = model.audio_dropout
        self.predict_layer = model.predict_layer[0]
        self.progress_scale = model.progress_scale
        self.args = model.args
        
    def _build_pm_rope_kwargs(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        cur_len: int,
        target_length: torch.Tensor,
        device: torch.device,
    ) -> dict:
        """Build PM-RoPE position arguments for decoder."""
        x_lens = encoder_attention_mask.sum(dim=1)
        max_len = encoder_hidden_states.shape[1]
        
        # Encoder position ids
        enc_pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
        denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
        encoder_position_ids = enc_pos / denom * self.progress_scale
        mask = enc_pos < x_lens[:, None]
        encoder_position_ids = encoder_position_ids.masked_fill(~mask, 0.0)
        
        # Decoder position ids
        est_total = target_length.float() + 1  # account for BOS
        base = torch.arange(cur_len, device=device, dtype=torch.float32).unsqueeze(0)
        decoder_position_ids = base / (est_total[:, None].clamp(min=2) - 1) * self.progress_scale
        
        return {
            "position_ids": decoder_position_ids,
            "pm_decoder_position_ids": decoder_position_ids,
            "pm_encoder_position_ids": encoder_position_ids,
        }
        
    def forward(
        self,
        prompt_tokens: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        target_length: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process audio prompt and return logits for next token.
        
        Args:
            prompt_tokens: [batch, prompt_len]
            encoder_hidden_states: [batch, enc_len, hidden]
            encoder_attention_mask: [batch, enc_len]
            target_length: [batch] - estimated total length
            
        Returns:
            logits: [batch, 1, vocab_size]
        """
        batch_size = prompt_tokens.shape[0]
        device = prompt_tokens.device
        
        # Prepend BOS token
        bos = torch.full(
            (batch_size, 1),
            self.args.empty_token,
            dtype=torch.long,
            device=device,
        )
        tokens = torch.cat([bos, prompt_tokens], dim=1)
        
        # Embed audio tokens
        embedded_y = self.audio_dropout(self.audio_embedding(tokens))
        cur_len = embedded_y.shape[1]
        
        decoder_attention_mask = torch.ones(
            (batch_size, cur_len), dtype=torch.long, device=device
        )
        
        # Build position kwargs
        use_pm_rope = getattr(self.args, "use_pm_rope", 1)
        pm_kwargs = {}
        if use_pm_rope:
            pm_kwargs = self._build_pm_rope_kwargs(
                encoder_hidden_states, encoder_attention_mask,
                cur_len, target_length, device
            )
        else:
            pm_kwargs["position_ids"] = None
            
        # Run decoder
        decoder_outputs = self.decoder(
            inputs_embeds=embedded_y,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            **pm_kwargs,
        )
        
        # Get logits for next token
        last_hidden = decoder_outputs.last_hidden_state[:, -1:, :]
        return self.predict_layer(last_hidden)


class DecoderStepWrapper(nn.Module):
    """Wrapper for single decoder step with KV cache."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.decoder = model.decoder_module
        self.audio_embedding = model.audio_embedding[0]
        self.audio_dropout = model.audio_dropout
        self.predict_layer = model.predict_layer[0]
        self.progress_scale = model.progress_scale
        self.args = model.args
        
    def forward(
        self,
        input_token: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        current_length: torch.Tensor,
        target_length: torch.Tensor,
        past_key_values: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Single autoregressive step.
        
        Returns:
            logits: [batch, 1, vocab_size]
            new_past_key_values: updated KV cache as flat tuple
        """
        device = input_token.device
        batch_size = input_token.shape[0]
        
        embedded = self.audio_dropout(self.audio_embedding(input_token))
        
        # Wrap tuple in TupleCache for compatibility with T5Gemma decoder
        cache = TupleCache(past_key_values)
        past_seq_len = cache.get_seq_length()
        
        # Use tensor arithmetic instead of .item() to avoid TracerWarning
        # attention_mask length = past_seq_len + 1 (for new token)
        attention_mask = torch.ones(
            (batch_size, past_seq_len + 1),
            dtype=torch.long,
            device=device
        )
        
        # Build position kwargs
        use_pm_rope = getattr(self.args, "use_pm_rope", 1)
        pm_kwargs = {}
        
        if use_pm_rope:
            x_lens = encoder_attention_mask.sum(dim=1)
            max_len = encoder_hidden_states.shape[1]
            
            enc_pos = torch.arange(max_len, device=device, dtype=torch.float32)[None, :]
            denom = (x_lens.clamp(min=2).to(torch.float32) - 1.0)[:, None]
            encoder_position_ids = enc_pos / denom * self.progress_scale
            mask = enc_pos < x_lens[:, None]
            encoder_position_ids = encoder_position_ids.masked_fill(~mask, 0.0)
            
            cur_len = current_length.float()
            est_total = target_length.float() + 1
            new_pos_value = (cur_len / (est_total.clamp(min=2) - 1) * self.progress_scale).clamp(max=self.progress_scale)
            pos_1 = new_pos_value.unsqueeze(-1)
            
            pm_kwargs = {
                "position_ids": pos_1,
                "pm_decoder_position_ids": pos_1,
                "pm_encoder_position_ids": encoder_position_ids,
            }
        else:
            pm_kwargs["position_ids"] = None
            
        decoder_outputs = self.decoder(
            inputs_embeds=embedded,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=cache,
            use_cache=True,
            **pm_kwargs,
        )
        
        logits = self.predict_layer(decoder_outputs.last_hidden_state)
        
        # Convert output cache back to flat tuple for ONNX
        new_cache = decoder_outputs.past_key_values
        if hasattr(new_cache, 'to_tuple'):
            new_past = new_cache.to_tuple()
        elif hasattr(new_cache, 'key_cache'):
            # DynamicCache format
            new_past = []
            for i in range(len(new_cache.key_cache)):
                new_past.append(new_cache.key_cache[i])
                new_past.append(new_cache.value_cache[i])
            new_past = tuple(new_past)
        else:
            new_past = new_cache
            
        return logits, new_past


class XCodec2DecoderWrapper(nn.Module):
    """Wrapper for XCodec2 decoder (codes -> audio)."""
    
    def __init__(self, codec: nn.Module):
        super().__init__()
        self.codec = codec
        
    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.ndim == 2:
            codes = codes.unsqueeze(1)
        return self.codec.decode_code(codes.long())
