# Simple Video/Audio Transcriber

Just run it and it works! Automatically chooses the best processing mode for your situation.

## Features

- 🧠 **Smart Auto-Mode** - program decides what's best
- 🔄 **Background processing** - works when laptop sleeps
- 📁 **Drop & Go** - add files and run
- 🚀 **Fast processing** with Whisper models

## Supported Formats

**Video**: MP4, AVI, MOV, MKV, WebM, FLV
**Audio**: MP3, WAV, M4A, AAC, FLAC, OGG

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

### Smart Auto-Mode (Just Run It!)

```bash
python main.py
```

**That's it!** The program automatically:

- ✅ **Detects files** in `input_media/` folder
- ✅ **Processes all files** and exits cleanly
- ✅ **Shows interactive menu** if no files found
- ✅ **Handles sleep/wake** during processing
- ✅ **Saves results** to `output_transcriptions/`

### Advanced Usage

```bash
# Process specific files
python main.py --files file1.mp3 file2.mp4

# Use different model
python main.py --files file.mp3 --model base

# Continuous daemon mode (watches for new files)
python main.py --daemon start --foreground

# Check daemon status
python main.py --daemon status

# Stop daemon
python main.py --daemon stop

# List available files
python main.py --list-files

# List available models
python main.py --list-models
```

### Directories

- `input_media/` - Place your audio/video files here
- `output_transcriptions/` - Transcriptions are saved here automatically
- `temp/` - Temporary files (auto-cleaned)

### Models

- **tiny** - Fastest, good quality (default)
- **base** - Good balance of speed and accuracy
- **small** - Better accuracy, slower
- **medium** - High accuracy, slower
- **large** - Best accuracy, slowest

## Examples

```bash
# Basic transcription
python main.py --files myfile.mp3

# Use different model
python main.py --files myfile.mp3 --model base

# Background processing
python main.py --daemon start --foreground
```

## Requirements

- Python 3.8+
- FFmpeg (install with `brew install ffmpeg` on macOS)
- Dependencies: `pip install -r requirements.txt`
