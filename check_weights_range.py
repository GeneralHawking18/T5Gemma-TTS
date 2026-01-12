import numpy as np

weights = np.load("/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/trt_weights/weights.npz")
for key in sorted(weights.files):
    w = weights[key]
    print(f"{key:<60} | shape={str(w.shape):<20} | range=[{w.min():.4f}, {w.max():.4f}] | std={w.std():.4f}")
