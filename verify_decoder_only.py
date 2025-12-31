import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM
import os
from dotenv import load_dotenv

load_dotenv()

def verify_decoder_only():
    model_name = "Aratako/T5Gemma-TTS-2b-2b-encoder-4bit"
    weights_path = "weights/t5gemma_decoder_only.bin"
    
    print(f"Loading config from {model_name}...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    print("Initializing model skeleton on meta device...")
    # Initialize empty model
    with torch.device("meta"):
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True)
        
    print(f"Loading decoder weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # We only want to load the decoder.
    # The keys in the bin file are likely "backbone.model.decoder..." or similar depending on export.
    # Let's inspect one key to be sure.
    first_key = next(iter(state_dict.keys()))
    print(f"Sample key from weights: {first_key}")

    # To load into the full model structure, we can just load_state_dict with strict=False
    # This will populate the decoder and leave encoder on meta/empty.
    print("Loading state dict into model (strict=False)...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    
    print(f"Missing keys (expected to include encoder): {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    
    # Verify encoder is still on meta (or empty)
    # Note: assign=True moves matched params to CPU, others stay on meta.
    
    print("\nPreparing dummy inputs for Decoder...")
    device = "cpu"
    # Ensure decoder is on CPU (should be from load_state_dict(assign=True))
    # Or explicitly move it if needed, but assign=True handles it for loaded weights.
    
    # Create dummy inputs
    B, T = 1, 10
    # Try to find hidden size from config
    hidden_dim = getattr(config, "d_model", getattr(config, "hidden_size", None))
    if hidden_dim is None and hasattr(config, "encoder"):
        hidden_dim = getattr(config.encoder, "d_model", getattr(config.encoder, "hidden_size", None))
    
    if hidden_dim is None:
        print("Warning: Could not find hidden size in config. Deriving from weights...")
        # Try to infer from a layer weight shape
        for k, v in state_dict.items():
            if "norm.weight" in k and len(v.shape) == 1:
                hidden_dim = v.shape[0]
                print(f"Derived hidden_dim={hidden_dim} from {k}")
                break
        if hidden_dim is None:
            hidden_dim = 2304  # fallback based on error message
            print(f"Using fallback hidden_dim={hidden_dim}")

    print(f"Using hidden_dim: {hidden_dim}")
    
    print(f"Using hidden_dim: {hidden_dim}")
    
    # Locate decoder first
    print("Locating decoder module...")
    decoder = None
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        decoder = model.model.decoder
    elif hasattr(model, "decoder"):
        decoder = model.decoder
    else:
        for name, child in model.named_children():
            if "decoder" in name:
                decoder = child
                break
    
    if decoder is None:
        if hasattr(model, "backbone") and hasattr(model.backbone, "model") and hasattr(model.backbone.model, "decoder"):
             decoder = model.backbone.model.decoder

    if decoder is None:
        print("Could not locate decoder module! Exiting.")
        return

    print("Materializing decoder module to CPU (resolves meta tensor issues)...")
    decoder.to_empty(device="cpu")

    print("Filtering and loading weights into decoder...")
    # Filter for decoder keys and strip prefix
    # Adjust prefix based on inspection. Previous run showed `backbone.model.decoder.norm.weight`
    # We want to load into `decoder` object. If `decoder` was `backbone.model.decoder`, then prefix is `backbone.model.decoder.`
    
    # Find the prefix that matches the decoder module path likely
    # But since we have the keys, we can just look for "mouse pointer" logic
    prefix = "backbone.model.decoder."
    
    local_state = {}
    loaded_count = 0
    for k, v in state_dict.items():
        if k.startswith(prefix):
            local_key = k[len(prefix):]
            local_state[local_key] = v
            loaded_count += 1
            
    print(f"Loaded {loaded_count} keys into local state dict for decoder.")
    if loaded_count == 0:
        print("WARNING: No keys matched prefix 'backbone.model.decoder.'. Checking for other patterns...")
        # Fallback: check if keys start with "decoder." or "model.decoder."
        pass

    missing, unexpected = decoder.load_state_dict(local_state, strict=False)
    print(f"Decoder load result - Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Decoder inputs: [B, T, hidden] - use float16 to match model weights
    decoder_inputs_embeds = torch.randn(B, T, hidden_dim, device=device, dtype=torch.float16)
    
    # Encoder hidden states (memory): [B, S, hidden]
    S = 20
    encoder_hidden_states = torch.randn(B, S, hidden_dim, device=device, dtype=torch.float16)
    
    # Attention masks
    decoder_attention_mask = torch.ones(B, 1, T, T, device=device, dtype=torch.float16) # minimal causal mask
    encoder_attention_mask = torch.ones(B, S, device=device).long()
    
    print("Running model.model.decoder forward pass...")
    try:
        # Accessing the internal decoder module directly
        # Structure is likely model.model.decoder or model.decoder depending on architecture class
        decoder = None
        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            decoder = model.model.decoder
        elif hasattr(model, "decoder"):
            decoder = model.decoder
            
        if decoder is None:
            print("Could not locate decoder module at standard paths. Searching children...")
            for name, child in model.named_children():
                print(f"Child: {name}")
                if "decoder" in name:
                    decoder = child
                    print(f"Found decoder at: {name}")
                    break
        
        if decoder is None:
            # Try deeper search
            if hasattr(model, "backbone") and hasattr(model.backbone, "model") and hasattr(model.backbone.model, "decoder"):
                decoder = model.backbone.model.decoder
                print("Found decoder at: model.backbone.model.decoder")

        if decoder is None:
             print("FINAL ATTEMPT: Checking keys in state dict again to guess path")
             print(f"Sample key: {next(iter(state_dict.keys()))}")
             return

        outputs = decoder(
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True
        )
        
        print("Success!")
        print(f"Output type: {type(outputs)}")
        if hasattr(outputs, "last_hidden_state"):
            print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_decoder_only()
