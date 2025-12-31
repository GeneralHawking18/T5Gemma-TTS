
import logging
import torch
from transformers import AutoConfig
import transformers.masking_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _custom_no_vmap_sdpa_mask(batch_size, cache_position, kv_length, kv_offset=0, mask_function=None, attention_mask=None, **kwargs):
    logger.info("Custom SDPA mask called!")
    device = cache_position.device
    query_idx = cache_position.unsqueeze(1)
    key_idx = torch.arange(kv_length, device=device).unsqueeze(0) + kv_offset
    causal_mask = query_idx >= key_idx
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        padding_mask = attention_mask.to(torch.bool)[:, None, None, :]
        causal_mask = causal_mask & padding_mask
    return causal_mask

def test_patch():
    print(f"Before patch: {transformers.masking_utils.sdpa_mask_recent_torch}")
    
    # Apply patch
    transformers.masking_utils.sdpa_mask = _custom_no_vmap_sdpa_mask
    transformers.masking_utils.sdpa_mask_recent_torch = _custom_no_vmap_sdpa_mask
    transformers.masking_utils.sdpa_mask_older_torch = _custom_no_vmap_sdpa_mask
    
    print(f"After patch: {transformers.masking_utils.sdpa_mask_recent_torch}")
    
    # Test if create_causal_mask uses it
    # We simulate arguments for create_causal_mask
    # input_shape, dtype, device
    try:
        mask = transformers.masking_utils.create_causal_mask(
            input_shape=(1, 10),
            dtype=torch.float32,
            device="cpu",
            past_key_values_length=0
        )
        print("create_causal_mask returned successfully")
    except Exception as e:
        print(f"create_causal_mask failed: {e}")

if __name__ == "__main__":
    test_patch()
