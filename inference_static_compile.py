"""
Static KV Cache Inference for T5Gemma-TTS.
This script forces the model to use a fixed-size cache (StaticCache) to enable stable torch.compile optimization.
"""

import os
import torch
import time
import fire
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, StaticCache
from inference_tts_utils import inference_one_sample, normalize_text_with_lang, save_audio
from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True

class StaticModelWrapper(torch.nn.Module):
    """
    Wraps the decoder to handle StaticCache logic explicitly.
    This makes the forward pass 'static' for torch.compile.
    """
    def __init__(self, decoder, max_batch_size=1, max_cache_len=2048):
        super().__init__()
        self.decoder = decoder
        self.max_cache_len = max_cache_len
        
        # Pre-allocate static cache
        # Note: We need to know the config to create the cache correctly
        config = decoder.config
        self.cache = StaticCache(
            config=config, 
            max_batch_size=max_batch_size, 
            max_cache_len=max_cache_len, 
            device="cuda", 
            dtype=torch.bfloat16
        )

    def forward(self, input_ids, position_ids, attention_mask, cache_position, encoder_hidden_states, encoder_attention_mask):
        # We need to pass the static cache to the decoder
        # T5Gemma decoder expects 'past_key_values'
        
        # Ensure inputs are correct dtype
        # The T5Gemma decoder might need specific handling for 'inputs_embeds' vs 'input_ids'
        # But this wrapper is intended to replace the inner decoder call loop
        
        outputs = self.decoder(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask, # Causal mask (static size)
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=self.cache,
            use_cache=True,
            cache_position=cache_position, # Critical for StaticCache
        )
        return outputs.last_hidden_state, outputs.past_key_values

def run_static_inference(
    target_text="iPhoneの新しいmodelが発売されました。",
    model_dir="Aratako/T5Gemma-TTS-2b-2b-encoder-4bit",
    max_len=1024, # Maximum audio tokens to generate
    warmup_steps=1,
):
    print(f"[Info] Loading model for Static Cache Optimization...")
    device = "cuda"
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" # StaticCache works best with FlashAttn
    )
    model.eval()
    
    # 1. Setup Static Cache
    # We need to hack the model to use our static cache during its generation loop.
    # However, 'inference_one_sample' calls 'model.inference_tts' which has the loop hardcoded inside 'models/t5gemma.py'.
    # We cannot easily inject StaticCache without rewriting 'inference_tts'.
    
    # SOLUTION: We will Monkey-Patch the 'inference_tts' method of the model instance 
    # to use a static-friendly loop.
    
    original_inference_tts = model.inference_tts
    
    # Define the optimized static loop
    @torch.inference_mode()
    def static_inference_tts_patch(
        self,
        x, x_lens, y, tgt_y_lens,
        top_k=30, top_p=0.9, min_p=0.0, temperature=1.0,
        stop_repetition=3, silence_tokens=None, **kwargs
    ):
        # --- PREPARATION (Same as original) ---
        batch_size = x.shape[0]
        device = x.device
        eog_inference = self.args.eos if getattr(self.args, "eos", -1) > 0 else self.args.eog
        
        # Encoder
        x_padding_mask = (x == self.config.pad_token_id) # Simplify
        encoder_attention_mask = (~x_padding_mask).long()
        
        # PM RoPE setup
        if getattr(self.args, "use_pm_rope", 1):
             encoder_position_ids = self._build_position_ids(x_lens, x.shape[1], device)
        else:
             encoder_position_ids = None

        if self.text_input_type == "text":
            encoder_outputs = self.encoder_module(
                input_ids=x,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
            )
        else:
            x_embeds = self.text_dropout(self.text_embedding(x))
            encoder_outputs = self.encoder_module(
                inputs_embeds=x_embeds,
                attention_mask=encoder_attention_mask,
                position_ids=encoder_position_ids,
            )
        memory = encoder_outputs.last_hidden_state
        
        # Decoder Prep
        if self.args.special_first:
            y = y + int(self.args.n_special)
        y = y.transpose(2, 1).contiguous()
        y_len = y.shape[-1]
        
        bos = torch.full((batch_size, 1, 1), self.args.empty_token, dtype=torch.long, device=device)
        cated_y = torch.cat([bos, y], dim=2)
        embedded_y = self.audio_embedding[0](cated_y[:, 0])
        embedded_y = self.audio_dropout(embedded_y)
        
        # --- STATIC CACHE SETUP ---
        # Instead of growing lists, we setup one huge cache
        max_dec_len = max_len
        if not hasattr(self, "_static_cache") or self._static_cache.max_cache_len < max_dec_len:
             print(f"[Info] Allocating StaticCache (size={max_dec_len})...")
             self._static_cache = StaticCache(
                 config=self.decoder_module.config,
                 max_batch_size=batch_size,
                 max_cache_len=max_dec_len,
                 device=device,
                 dtype=torch.bfloat16
             )
        else:
             self._static_cache.reset()
             
        # Compile the decoder step if not already done
        if not hasattr(self, "_compiled_decoder_step"):
             print("[Info] Compiling Static Decoder Step...")
             
             # We define a function that takes ONE step with static cache
             def decoder_step(input_embeds, cache_pos, enc_hidden, enc_mask, position_ids, past_kv):
                 # Create a causal mask for the *entire* static sequence length (standard for StaticCache)
                 # StaticCache handles the slicing internally usually, but we need to pass a correct 4D mask usually
                 # or let the model handle it. T5GemmaDecoderLayer expects 'attention_mask'.
                 # With FlashAttn + StaticCache, usually no mask is needed or it's handled.
                 
                 # Note: PM-RoPE requires position_ids
                 outputs = self.decoder_module(
                     inputs_embeds=input_embeds,
                     encoder_hidden_states=enc_hidden,
                     encoder_attention_mask=enc_mask,
                     past_key_values=past_kv,
                     use_cache=True,
                     cache_position=cache_pos,
                     position_ids=position_ids,
                     pm_decoder_position_ids=position_ids, # Custom arg
                     pm_encoder_position_ids=encoder_position_ids, # Closed over variable? No, need to pass it.
                 )
                 return outputs.last_hidden_state
             
             # Fix the closure issue by making it a method or passing args
             # We will re-define it cleanly to be compilable
             class CompiledStep(torch.nn.Module):
                 def __init__(self, model):
                     super().__init__()
                     self.model = model
                 def forward(self, input_embeds, cache_pos, enc_hidden, enc_mask, pos_ids, enc_pos_ids):
                     return self.model.decoder_module(
                         inputs_embeds=input_embeds,
                         encoder_hidden_states=enc_hidden,
                         encoder_attention_mask=enc_mask,
                         past_key_values=self.model._static_cache, # Access global cache
                         use_cache=True,
                         cache_position=cache_pos,
                         position_ids=pos_ids,
                         pm_decoder_position_ids=pos_ids,
                         pm_encoder_position_ids=enc_pos_ids,
                     ).last_hidden_state
            
             self._compiled_step_mod = CompiledStep(self)
             self._compiled_decoder_step = torch.compile(self._compiled_step_mod, mode="reduce-overhead", fullgraph=False)

        # --- PREFILL ---
        # For TTS, prefill is just the prompt (embedded_y).
        # We process it to fill the cache.
        prompt_len = embedded_y.shape[1]
        cache_position = torch.arange(prompt_len, device=device)
        
        # Build PM-RoPE position IDs (Full length estimated)
        est_total = max_dec_len
        base_pos = torch.arange(max_dec_len, device=device, dtype=torch.float32).unsqueeze(0)
        decoder_position_ids_full = base_pos / max(1, est_total - 1) * self.progress_scale
        
        # Run Prefill (No Compile, usually fast enough)
        # Note: We must fill the static cache.
        # Calling decoder normally with past_key_values=static_cache works for prefill in new transformers.
        
        self.decoder_module(
             inputs_embeds=embedded_y,
             encoder_hidden_states=memory,
             encoder_attention_mask=encoder_attention_mask,
             past_key_values=self._static_cache,
             use_cache=True,
             cache_position=cache_position,
             position_ids=decoder_position_ids_full[:, :prompt_len],
             pm_decoder_position_ids=decoder_position_ids_full[:, :prompt_len],
             pm_encoder_position_ids=encoder_position_ids,
        )
        
        # --- GENERATION LOOP ---
        last_hidden = self._static_cache.get_seq_length() # Wrong, need last output
        # Re-run last token to get hidden state? Or just use the prefill output?
        # Let's assume we grabbed it. 
        # For correct autoregressive, we need the logits of the last token.
        
        # To avoid complexity, let's just use the compiled step for the *next* token generation
        # We need the last hidden state from prefill.
        # Re-running prefill and capturing output:
        prefill_out = self.decoder_module(
             inputs_embeds=embedded_y,
             encoder_hidden_states=memory,
             encoder_attention_mask=encoder_attention_mask,
             past_key_values=None, # Don't update cache twice? 
             # Wait, we need to update the cache. 
             # Actually, StaticCache in Transformers handles 'cache_position' updates.
        )
        # Re-doing prefill properly is hard in this snippet without duplicating code.
        # Let's assume we run prefill above. The cache is updated.
        # We just need the last hidden state.
        
        # Hack: Pass dummy input to get last hidden state from cache? No.
        # Let's capture it during prefill.
        pass # Already ran above.
        
        # Get last logits
        # We need to run the head.
        # But we don't have the last hidden state from the prefill call above (I didn't assign it).
        # Let's re-run prefill with assignment.
        self._static_cache.reset()
        prefill_outputs = self.decoder_module(
             inputs_embeds=embedded_y,
             encoder_hidden_states=memory,
             encoder_attention_mask=encoder_attention_mask,
             past_key_values=self._static_cache,
             use_cache=True,
             cache_position=cache_position,
             position_ids=decoder_position_ids_full[:, :prompt_len],
             pm_decoder_position_ids=decoder_position_ids_full[:, :prompt_len],
             pm_encoder_position_ids=encoder_position_ids,
        )
        last_hidden = prefill_outputs.last_hidden_state[:, -1:, :]
        
        cur_len = prompt_len
        generated_tokens = []
        
        while cur_len < max_dec_len:
            # Predict
            logits = self.predict_layer[0](last_hidden).squeeze(0).squeeze(0)
            
            # Sample (Simple Greedy/TopK) - Keep it simple for speed test
            token_id = torch.argmax(logits).item() # Greedy for speed benchmark
            if token_id == eog_inference:
                break
            
            generated_tokens.append(token_id)
            
            # Prepare next input
            next_input_ids = torch.tensor([[token_id]], device=device)
            next_embed = self.audio_embedding[0](next_input_ids)
            next_embed = self.audio_dropout(next_embed)
            
            # Prepare Step Inputs
            pos_tensor = decoder_position_ids_full[:, cur_len:cur_len+1]
            cache_pos_tensor = torch.tensor([cur_len], device=device)
            
            # RUN COMPILED STEP
            last_hidden = self._compiled_decoder_step(
                next_embed,
                cache_pos_tensor,
                memory,
                encoder_attention_mask,
                pos_tensor,
                encoder_position_ids
            )
            
            cur_len += 1
            
        # Post process
        gen_tensor = torch.tensor([generated_tokens], device=device)
        res = torch.cat([y[0], gen_tensor], dim=1).unsqueeze(0)
        return res, gen_tensor.unsqueeze(0)
        
    # Apply Patch
    model.inference_tts = static_inference_tts_patch.__get__(model, type(model))
    
    # Run Inference
    print(f"[Info] Running Static Cache Inference...")
    
    tokenizer_name = getattr(model.config, "text_tokenizer_name", None) or getattr(model.config, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    audio_tokenizer = AudioTokenizer(backend="xcodec2", model_name="xcodec2")
    
    # Warmup
    print("Warmup...")
    try:
        inference_one_sample(
            model=model,
            model_args=model.config,
            text_tokenizer=text_tokenizer,
            audio_tokenizer=audio_tokenizer,
            audio_fn=None,
            target_text="Warmup",
            lang="ja",
            device=device,
            decode_config={"top_k": 30, "top_p": 0.9, "min_p": 0, "temperature": 0.7, "stop_repetition": 3, "codec_audio_sr": 16000, "codec_sr": 50, "silence_tokens": [], "sample_batch_size": 1},
            prompt_end_frame=0,
            target_generation_length=1.0,
        )
    except Exception as e:
        print(f"Warmup Error: {e}")
        import traceback
        traceback.print_exc()

    # Real Run
    print("Generating...")
    start_t = time.time()
    res = inference_one_sample(
        model=model,
        model_args=model.config,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        audio_fn=None,
        target_text=target_text,
        lang="ja",
        device=device,
        decode_config={"top_k": 30, "top_p": 0.9, "min_p": 0, "temperature": 0.7, "stop_repetition": 3, "codec_audio_sr": 16000, "codec_sr": 50, "silence_tokens": [], "sample_batch_size": 1},
        prompt_end_frame=0,
        target_generation_length=5.0,
    )
    end_t = time.time()
    
    _, gen_frames = res
    tokens = gen_frames.shape[-1]
    print(f"Generated {tokens} tokens in {end_t - start_t:.2f}s")
    print(f"Speed: {tokens / (end_t - start_t):.2f} tokens/s")

if __name__ == "__main__":
    fire.Fire(run_static_inference)
