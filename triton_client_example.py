import tritonclient.http as httpclient
import numpy as np
import soundfile as sf
import argparse


def main(text):
    try:
        # Increase timeout to 300 seconds for model loading
        client = httpclient.InferenceServerClient(
            "localhost:8000", connection_timeout=300.0, network_timeout=300.0
        )

        # Prepare input
        # Shape [1, 1] for batch size 1, dim 1
        inputs = [httpclient.InferInput("TEXT", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[text.encode("utf-8")]], dtype=object))

        # Inference
        print(f"Sending request: {text}")
        result = client.infer("t5gemma_tts", inputs)
        audio = result.as_numpy("AUDIO_WAVEFORM").flatten()

        # Save
        sf.write("output.wav", audio, 44100)
        print("Saved to output.wav")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="こんにちは、これはテストです。")
    args = parser.parse_args()
    main(args.text)
