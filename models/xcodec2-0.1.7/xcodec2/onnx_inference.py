from onnxruntime import InferenceSession


import soundfile as sf
import time
import numpy as np



ONNX_PATH="/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/models/xcodec2-0.1.7/vocoder.onnx"
OUTPUT_SR = 44100 
VQ_CODE_CACHE_PATH="/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/models/xcodec2-0.1.7/vq_code_cache.npy"
OUTPUT_AUDIO="./recon_onnx.wav"

inference_session = InferenceSession(
    providers=["CUDAExecutionProvider"],
    path_or_bytes=ONNX_PATH,
)
vq_code = np.load(VQ_CODE_CACHE_PATH)

start = time.time()
recon = inference_session.run(
    output_names=['recon_audio'],
    input_feed={
        "vq_code": vq_code
    }
)[0]


recon_wav = recon[0, 0]
sf.write(OUTPUT_AUDIO, recon_wav, OUTPUT_SR)

print("Xong!")
print(f"Output (44.1kHz): {OUTPUT_AUDIO}")
print("Inference time: ", time.time() - start)
