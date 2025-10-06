from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nextai",
    version="1.0.0",
    author="Sanyam Suyal",
    author_email="contact@nextbench.com",
    description="An AI model for educational and career guidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SanyamSuyal/NextAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "sentencepiece>=0.1.99",
        "huggingface-hub>=0.15.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "training": [
            "wandb>=0.15.0",
            "tensorboard>=2.13.0",
        ],
        "api": [
            "flask>=2.3.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nextai=nextai.cli:main",
        ],
    },
)
