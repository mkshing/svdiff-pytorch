from setuptools import find_packages, setup


setup(
    name="svdiff-pytorch",
    version="0.2.0",
    author="Makoto Shing",
    url="https://github.com/mkshing/svdiff-pytorch",
    description="Implementation of 'SVDiff: Compact Parameter Space for Diffusion Fine-Tuning'",
    install_requires=[
    "diffusers==0.14.0",
    "accelerate",
    "torchvision",
    "safetensors",
    "transformers>=4.25.1",
    "ftfy",
    "tensorboard",
    "Jinja2",
    "einops",
    "wandb"
    ],
    packages=find_packages(exclude=("examples", "build")),
    license = 'MIT',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)