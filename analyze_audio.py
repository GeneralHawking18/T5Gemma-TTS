
import wave
import struct
import os
import sys

filepath = "/hdd/chungha/hdd-data/Mazii_AI-casual-TTS/T5Gemma-TTS/outputs/test_hybrid_tts.wav"

try:
    if not os.path.exists(filepath):
        print("File not found.")
        sys.exit(1)
        
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        comp_type = wf.getcomptype()
        comp_name = wf.getcompname()
        
        print(f"Channels: {n_channels}")
        print(f"Sample Width: {sampwidth}")
        print(f"Frame Rate: {framerate}")
        print(f"Frames: {n_frames}")
        print(f"Compression: {comp_type} ({comp_name})")
        
        if n_frames == 0:
            print("File is empty (no frames).")
            sys.exit(0)
            
        data = wf.readframes(n_frames)
        
        # Parse data based on sample width
        if sampwidth == 2:
            fmt = f"<{n_frames * n_channels}h"
            samples = struct.unpack(fmt, data)
        elif sampwidth == 1:
            fmt = f"<{n_frames * n_channels}B"
            samples = struct.unpack(fmt, data)
        else:
            print(f"Unsupported sample width: {sampwidth}")
            sys.exit(0)
            
        min_val = min(samples)
        max_val = max(samples)
        mean_val = sum(samples) / len(samples)
        
        print(f"Min: {min_val}, Max: {max_val}")
        print(f"Mean: {mean_val}")
        
        if min_val == 0 and max_val == 0:
            print("Audio is PURE SILENCE.")
        elif min_val == max_val:
            print("Audio is FLAT (constant value).")
        
except Exception as e:
    print(f"Error reading file: {e}")
