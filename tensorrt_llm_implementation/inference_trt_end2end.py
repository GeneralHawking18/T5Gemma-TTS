import tensorrt as trt
import torch
import numpy as np
import argparse
import os
import time
import soundfile as sf
from transformers import AutoTokenizer

# CRITICAL: Import tensorrt_llm to register custom plugins (like CumsumLastDim)
import tensorrt_llm 

# Import Audio Tokenizer from existing codebase
import sys
sys.path.append(os.getcwd())

try:
    from data.tokenizer import AudioTokenizer
except ImportError:
    print("[Warn] Could not import AudioTokenizer (likely missing torchaudio). Audio decoding will be skipped.")
    AudioTokenizer = None

def load_engine(engine_path):
    print(f"[TRT] Loading engine: {engine_path}")
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Engine not found: {engine_path}")
        
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

class TRTEncoder:
    def __init__(self, engine_path):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        self.context.set_input_shape("input_ids", (batch_size, seq_len))
        self.context.set_input_shape("attention_mask", (batch_size, seq_len))
        
        hidden_size = 2304
        output = torch.empty((batch_size, seq_len, hidden_size), dtype=torch.float16, device="cuda")
        
        self.context.set_tensor_address("input_ids", input_ids.data_ptr())
        self.context.set_tensor_address("attention_mask", attention_mask.data_ptr())
        self.context.set_tensor_address("encoder_hidden_states", output.data_ptr())
        
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return output

class TRTDecoder:
    def __init__(self, engine_path):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        
    def forward(self, input_ids, encoder_hidden_states, position_ids, encoder_position_ids, encoder_attention_mask):
        batch, dec_seq = input_ids.shape
        _, enc_seq, hidden = encoder_hidden_states.shape
        
        self.context.set_input_shape("input_ids", (batch, dec_seq))
        self.context.set_input_shape("encoder_hidden_states", (batch, enc_seq, hidden))
        self.context.set_input_shape("position_ids", (batch, dec_seq))
        self.context.set_input_shape("encoder_position_ids", (batch, enc_seq))
        self.context.set_input_shape("encoder_attention_mask", (batch, enc_seq))
        
        output = torch.empty((batch, dec_seq, hidden), dtype=torch.float16, device="cuda")
        
        self.context.set_tensor_address("input_ids", input_ids.data_ptr())
        self.context.set_tensor_address("encoder_hidden_states", encoder_hidden_states.data_ptr())
        self.context.set_tensor_address("position_ids", position_ids.data_ptr())
        self.context.set_tensor_address("encoder_position_ids", encoder_position_ids.data_ptr())
        self.context.set_tensor_address("encoder_attention_mask", encoder_attention_mask.data_ptr())
        self.context.set_tensor_address("output", output.data_ptr())
        
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="こんにちは、これはテストです。")
    parser.add_argument("--encoder_engine", type=str, default="/app/trt_weights/encoder.engine")
    parser.add_argument("--decoder_engine", type=str, default="/app/trt_weights/t5gemma_decoder_new.engine")
    parser.add_argument("--weights_path", type=str, default="/app/trt_weights/weights.npz")
    parser.add_argument("--tokenizer_path", type=str, default="Aratako/T5Gemma-TTS-2b-2b")
    parser.add_argument("--output_wav", type=str, default="output_trt.wav")
    args = parser.parse_args()
    
    # 1. Load Models
    print("[Info] Loading models...")
    if not os.path.exists(args.encoder_engine):
        print(f"[Wait] Encoder engine not found at {args.encoder_engine}. Waiting for build to finish...")
        # Simple wait loop
        for _ in range(60): # Wait up to 10 mins (60 * 10s)
            if os.path.exists(args.encoder_engine):
                print("Found engine!")
                break
            time.sleep(10)
            print(".", end="", flush=True)
        else:
            raise FileNotFoundError("Encoder engine build timed out or failed.")

    enc = TRTEncoder(args.encoder_engine)
    
    # Wait for Decoder engine
    if not os.path.exists(args.decoder_engine):
        print(f"[Wait] Decoder engine not found at {args.decoder_engine}. Waiting for build to finish...")
        for _ in range(120): # Wait up to 20 mins
            if os.path.exists(args.decoder_engine):
                print("Found decoder engine!")
                # Give it a moment to finish writing
                time.sleep(5) 
                break
            time.sleep(10)
            print(".", end="", flush=True)
        else:
             print(f"[Warn] Decoder engine {args.decoder_engine} not found after waiting. Please build it first.")
             return

    dec = TRTDecoder(args.decoder_engine)
    
    # Load Projection Layer
    print(f"[Info] Loading projection weights from {args.weights_path}")
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Weights file not found: {args.weights_path}")
    
    weights = np.load(args.weights_path)
    if "audio_embedding.weight" in weights:
        embed_weight = torch.from_numpy(weights["audio_embedding.weight"]).to(dtype=torch.float16, device="cuda")
    elif "embed_tokens.weight" in weights:
        embed_weight = torch.from_numpy(weights["embed_tokens.weight"]).to(dtype=torch.float16, device="cuda")
    else:
        raise KeyError("Could not find embedding weight in weights.npz")
        
    print(f"  - Embedding shape: {embed_weight.shape}") # [Vocab, Hidden]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    try:
        audio_tokenizer = AudioTokenizer(backend="xcodec2", model_name="xcodec2", device="cuda")
    except Exception as e:
        print(f"[Warn] Failed to load audio tokenizer: {e}")
        audio_tokenizer = None

    # 2. Preprocess Text
    print(f"[Info] Input text: {args.text}")
    input_ids = tokenizer.encode(args.text, return_tensors="pt").long().cuda()
    attention_mask = torch.ones_like(input_ids).long().cuda()
    
    # 3. Encoder Inference
    print("[Info] Running Encoder...")
    t0 = time.time()
    encoder_hidden_states = enc.forward(input_ids, attention_mask)
    t_enc = time.time() - t0
    print(f"  - Encoder time: {t_enc*1000:.2f}ms")
    
    # 4. Decoder Setup
    batch_size = 1
    enc_seq_len = encoder_hidden_states.shape[1]
    
    encoder_position_ids = torch.linspace(0, 1, enc_seq_len, dtype=torch.float32).unsqueeze(0).cuda()
    encoder_attention_mask = attention_mask.clone() # already long
    
    # Start token (assuming 0 for now, check config)
    decoder_input_ids = torch.tensor([[0]], dtype=torch.long).cuda()
    
    # 5. Generation Loop
    print("[Info] Running Decoder Generation...")
    max_new_tokens = 512 # Generate up to ~10s of audio
    generated_tokens = []
    
    t0 = time.time()
    for i in range(max_new_tokens):
        dec_seq_len = decoder_input_ids.shape[1]
        
        # Simple linear position IDs for test
        position_ids = torch.linspace(0, i/1000.0, dec_seq_len, dtype=torch.float32).unsqueeze(0).cuda()
        
        # Run Decoder
        hidden_states = dec.forward(
            decoder_input_ids, 
            encoder_hidden_states, 
            position_ids, 
            encoder_position_ids, 
            encoder_attention_mask
        )
        
        # Project last token: [Batch, 1, Hidden] @ [Hidden, Vocab] -> [Batch, 1, Vocab]
        last_hidden = hidden_states[:, -1:, :]
        logits = torch.matmul(last_hidden, embed_weight.t())
        
        # Greedy Sample
        next_token = torch.argmax(logits, dim=-1).int() # [Batch, 1]
        token_id = next_token.item()
        
        # Check EOS (assuming 1 is EOS, check config)
        if token_id == 1:
            print(f"  - Hit EOS at step {i}")
            break
            
        generated_tokens.append(token_id)
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
        
        if i % 50 == 0:
            print(f"  Step {i}: {token_id}", end="\r")
            
    t_gen = time.time() - t0
    print(f"\n  - Generation time: {t_gen:.2f}s ({len(generated_tokens)/t_gen:.2f} tok/s)")
    
    # 6. Save Audio
    if audio_tokenizer and generated_tokens:
        print("[Info] Decoding audio...")
        # Reshape to [1, 1, Seq]
        tokens_tensor = torch.tensor(generated_tokens, device="cuda").reshape(1, 1, -1)
        audio = audio_tokenizer.decode(tokens_tensor)
        
        # Save
        path = args.output_wav
        sf.write(path, audio[0].cpu().numpy(), 16000) # Assuming 16k
        print(f"[Success] Saved audio to {path}")
    else:
        print(f"\n[FINAL OUTPUT] Generated Tokens: {generated_tokens}")
        print("[Info] Audio decoding skipped (tokenizer missing).")

if __name__ == "__main__":
    main()