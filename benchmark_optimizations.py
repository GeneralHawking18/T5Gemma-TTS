"""
Head-to-Head Benchmark: No Compile vs. Compile (MaxAutotune)
"""
import subprocess
import re
import sys
import time
import os

# The two heavyweights
TEST_CASES = [
    {
        "name": "No Compile (Baseline)", 
        "flags": ["--compile_mode", "None", "--use_flash_attn", "True", "--warmup_steps", "0"]
    },
    {
        "name": "Compile (MaxAutotune)", 
        "flags": ["--compile_mode", "max-autotune-no-cudagraphs", "--use_flash_attn", "True", "--warmup_steps", "1"]
    },
]

TARGET_TEXT = "iPhoneの新しいmodelが発売されました。"

def parse_speed(output):
    # Split lines and find the relevant one
    for line in output.split("\n"):
        if "[Speed]" in line:
            # Remove potential ANSI codes (if any)
            clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            # Match number
            match = re.search(r"Speed]\s+([\d\.]+)\s+tokens/s", clean_line)
            if match:
                return float(match.group(1))
    return 0.0

def run_test(case):
    print(f"\n{'='*60}")
    print(f"Running: {case['name']}")
    print(f"{ '='*60}")
    
    cmd = [
        sys.executable, "inference_4bit_gpu_optimized.py",
        "--target_text", TARGET_TEXT,
    ] + case["flags"]
    
    # Use venv python
    if not sys.executable.endswith("venv/bin/python"):
         venv_python = "./.venv/bin/python"
         if os.path.exists(venv_python):
             cmd[0] = venv_python

    try:
        env = dict(os.environ)
        if "None" in case["flags"]:
             env["DISABLE_COMPILE"] = "1"
        else:
             env["DISABLE_COMPILE"] = "0"
        
        print("...Running inference...")
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        
        speed = parse_speed(result.stdout)
        
        # Validation print
        if speed == 0.0:
            print("[ERROR] Could not parse speed. Raw output sample:")
            print(result.stdout[-500:])
        else:
            print(f"-> Result: {speed} tokens/s")
            
        return speed

    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return 0.0

def main():
    results = []
    print("Starting Comparison...\n")
    
    for case in TEST_CASES:
        speed = run_test(case)
        results.append((case["name"], speed))
        
    print("\n\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{ 'Configuration':<25} | { 'Speed':<15} | { 'Gain'}")
    print("-" * 60)
    
    baseline = results[0][1]
    
    for name, speed in results:
        gain = f"{(speed/baseline - 1)*100:+.1f}%" if baseline > 0 else "N/A"
        if name == results[0][0]: gain = "Baseline"
        print(f"{name:<25} | {speed:<15} | {gain}")
    print("-" * 60)

if __name__ == "__main__":
    main()
