import tensorrt_llm
from modeling import T5GemmaDecoderTRT

class Config:
    vocab_size = 100
    hidden_size = 64
    d_kv = 32
    d_ff = 128
    num_decoder_layers = 1
    num_attention_heads = 2
    dtype = "float16"

c = Config()
model = T5GemmaDecoderTRT(c)
print("Model created successfully")
for name, _ in model.named_parameters():
    print(name)