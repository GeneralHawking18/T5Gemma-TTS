import os
import torch
import soundfile as sf
import librosa
from modeling_xcodec2 import XCodec2Model

# ==========================================
# 1. Cấu hình
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tên model mới (HuggingFace ID)
model_path = "NandemoGHS/Anime-XCodec2-44.1kHz-v2"

# File đầu vào và đầu ra
input_audio = "/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/models/xcodec2-0.1.7/input_audio/sample.wav"  # Thay bằng file của bạn
output_audio = "output_44k.wav"

# ==========================================
# 2. Load Model
# ==========================================
print(f"Đang tải model {model_path}...")
# Model này tự động nhận config từ thư viện custom bạn vừa cài
model = XCodec2Model.from_pretrained(model_path)
model.eval().to(device)

# ==========================================
# 3. Xử lý Audio
# ==========================================
# QUAN TRỌNG: Encoder của model này VẪN LÀ 16kHz.
# Ta phải ép input về 16kHz trước khi đưa vào.
print("Đang xử lý input...")
wav, sr = librosa.load(input_audio, sr=16000) 

# Chuyển stereo -> mono (nếu cần)
if wav.ndim == 2:
    wav = wav.mean(axis=1)

# Chuyển sang Tensor
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).to(device)

# ==========================================
# 4. Inference
# ==========================================
print("Đang chạy Inference (Upsampling lên 44.1kHz)...")
with torch.inference_mode():
    # B1: Nén (Encode) - Model nhìn thấy input 16kHz
    vq_code = model.encode_code(input_waveform=wav_tensor)
    print(vq_code)
    
    # B2: Giải nén (Decode) - Model bung ra output 44.1kHz
    # Nhờ bản custom lib, decoder tự biết dùng UpsamplerBlock mới
    recon = model.decode_code(vq_code)

# ==========================================
# 5. Lưu kết quả
# ==========================================
# Output thực tế của model này là 44100Hz
output_sr = 44100 

recon_wav = recon[0, 0].cpu().numpy()
sf.write(output_audio, recon_wav, output_sr)

print("Xong!")
print(f"Input (16kHz) : {input_audio}")
print(f"Output (44.1kHz): {output_audio}")