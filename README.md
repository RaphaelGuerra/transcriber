# Audio/Video Transcriber

Last updated: 2026-04-05

## Table of Contents

<!-- TOC start -->
- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Run Locally](#run-locally)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Status & Learnings](#status--learnings)
- [License](#license)
<!-- TOC end -->

[![Lint](https://github.com/RaphaelGuerra/transcriber/actions/workflows/readme-lint.yml/badge.svg)](https://github.com/RaphaelGuerra/transcriber/actions/workflows/readme-lint.yml)
[![Security](https://github.com/RaphaelGuerra/transcriber/actions/workflows/security.yml/badge.svg)](https://github.com/RaphaelGuerra/transcriber/actions/workflows/security.yml)

Small tool that transcribes audio and video files with an easy “drop & go”
workflow.

This is a portfolio side project — exploring auto‑mode jobs, background
processing, and a simple CLI around Whisper models. Not a production tool.

## What It Does

- Auto‑mode finds media in `input_media/` and guides you through selection
- Lets you pick a Whisper model (tiny/base/small/medium/large)
- Supports common audio/video formats, including exported iPhone Voice Memos `.m4a`
- Normalizes audio before transcription and auto-chunks long recordings for more reliable Whisper runs
- Saves TXT (default) or SRT/VTT to `output_transcriptions/`
- Optional background daemon mode to watch for new files

## How It Works

- Uses FFmpeg to extract/process audio when needed
- Runs Whisper transcription with the selected model
- Presents interactive prompts and progress; exits cleanly when done

## Run Locally

Prerequisites: Python 3.8+ and FFmpeg (including `ffprobe`)

```bash
pip install -r requirements.txt
python3 main.py
```

Advanced:

- Process specific files: `python3 main.py --files file1.mp3 file2.mp4`
- Choose model: `python3 main.py --files file.m4a --model base`
- Daemon mode: `python3 main.py --daemon start --foreground`

## Usage

```bash
# Interactive mode (auto-detects media in input_media/)
python3 main.py

# One-shot transcription with the "small" Whisper model
python3 main.py --files input_media/clip.m4a --model small

# Start daemon watcher and stay in foreground (debug)
python3 main.py --daemon start --foreground

# Export subtitles directly
python3 main.py --files input_media/clip.m4a --format srt
```

### Quick Checks

```bash
ffmpeg -version
ffprobe -version
python3 -m unittest discover -s tests -v
```

### Voice Memos Workflow

- Export a recording from the iPhone Voice Memos app as `.m4a`
- Place it in `input_media/` or pass it with `--files`
- The app probes the file with `ffprobe`, converts it to mono 16 kHz WAV, and automatically chunks long recordings before merging the final transcript back into one TXT/SRT/VTT output

## Tech Stack

- Python, FFmpeg
- OpenAI Whisper models

## Status & Learnings

- Functional prototype to practice job scheduling, CLI ergonomics, and media
  handling
- Next ideas: timestamps tuning, language auto‑detect

## License

All rights reserved. Personal portfolio project — not for production use.
