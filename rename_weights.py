import os
import shutil
try:
    if os.path.exists("weights/t5gemma_decoder_only.bin"):
        shutil.copy("weights/t5gemma_decoder_only.bin", "weights/pytorch_model.bin")
        print("Copied to pytorch_model.bin")
    else:
        print("Source not found")
except Exception as e:
    print(e)
