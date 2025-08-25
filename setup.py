"""
Setup configuration for Fast Video/Audio Transcriber
"""

import os

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A streamlined, high-performance tool for transcribing video and audio files using OpenAI's Whisper model."


# Read requirements
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="fast-transcriber",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A streamlined, high-performance tool for transcribing video and audio files using OpenAI's Whisper model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fast-transcriber",
    packages=find_packages(),
    py_modules=["transcriber_core", "main", "cli", "config", "daemon", "file_handler", "logger"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "transcriber=main:main",
        ],
    },
    include_package_data=True,
    keywords="whisper transcription audio video speech-to-text ai ml",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/fast-transcriber/issues",
        "Source": "https://github.com/yourusername/fast-transcriber",
    },
)
