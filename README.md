# Audio/Video Transcriber

Last updated: 2025-11-03

## Table of Contents

<!-- TOC start -->
- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [Run Locally](#run-locally)
- [Tech Stack](#tech-stack)
- [Status & Learnings](#status-learnings)
- [License](#license)
<!-- TOC end -->

Small tool that transcribes audio and video files with an easy “drop & go” workflow.

This is a portfolio side project — exploring auto‑mode jobs, background processing, and a simple CLI around Whisper models. Not a production tool.

## What It Does
- Auto‑mode finds media in `input_media/` and guides you through selection
- Lets you pick a Whisper model (tiny/base/small/medium/large)
- Supports common audio/video formats; saves results to `output_transcriptions/`
- Optional background daemon mode to watch for new files

## How It Works
- Uses FFmpeg to extract/process audio when needed
- Runs Whisper transcription with the selected model
- Presents interactive prompts and progress; exits cleanly when done

## Run Locally
Prerequisites: Python 3.8+ and FFmpeg

```bash
pip install -r requirements.txt
python main.py
```

Advanced:
- Process specific files: `python main.py --files file1.mp3 file2.mp4`
- Choose model: `python main.py --files file.mp3 --model base`
- Daemon mode: `python main.py --daemon start --foreground`

## Tech Stack
- Python, FFmpeg
- OpenAI Whisper models

## Status & Learnings
- Functional prototype to practice job scheduling, CLI ergonomics, and media handling
- Next ideas: SRT/VTT export options, timestamps tuning, language auto‑detect

## License
All rights reserved. Personal portfolio project — not for production use.
