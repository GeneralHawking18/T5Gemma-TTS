from setuptools import setup, find_packages

setup(
    name="xcodec2",
    version="0.1.7",
    description="A library for XCodec2 model.",
    author="Zhen Ye",
    author_email="zhenye213@gmail.com",
    url="https://huggingface.co/HKUST-Audio/xcodec2",
    packages=find_packages(exclude=["tests*", "docs*"]),
    install_requires=[
        "numpy>=2.0.2",
        "einops==0.8.0",
        "torch>=2.5.0",
        "torchao>=0.5.0",
        "torchaudio>=2.5.0",
        "torchtune>=0.3.1",
        "transformers>=4.45.2",
        "vector-quantize-pytorch==1.17.8",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
