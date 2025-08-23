# Fast Video/Audio Transcriber

A streamlined, high-performance tool for transcribing video and audio files using OpenAI's Whisper model. Designed for speed, efficiency, and ease of use with both sequential and parallel processing options.

## Features

- 🚀 **High Performance**: Optimized for speed with multiprocessing support
- 🎯 **Smart Processing**: Files are sorted by size for optimal completion time
- 📊 **Real-time Progress**: Live progress tracking with detailed statistics
- 🔧 **Flexible Models**: Support for all Whisper model sizes (tiny to large)
- 📁 **Batch Processing**: Handle multiple files efficiently
- 💾 **Memory Optimized**: Intelligent memory management and garbage collection
- 🎨 **Interactive CLI**: User-friendly command-line interface
- 📝 **Resume Support**: Continue interrupted transcriptions

## Supported Formats

**Video**: MP4, AVI, MOV, MKV, WebM, FLV
**Audio**: MP3, WAV, M4A, AAC, FLAC, OGG

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd transcriber
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python transcriber.py
   ```

## Usage

### Interactive Mode (Default)

Simply run the script and follow the interactive prompts:

```bash
python transcriber.py
```

The interface will guide you through:
- Selecting files from the `input_media/` directory
- Choosing a Whisper model
- Configuring processing options (parallel/sequential)
- Viewing completed transcriptions

### Directory Structure

```
transcriber/
├── input_media/          # Place your media files here
├── output_transcriptions/ # Transcriptions are saved here
├── temp/                 # Temporary files (auto-cleaned)
├── transcriber.py        # Main application
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Model Options

| Model | Speed | Accuracy | Size | Use Case |
|-------|-------|----------|------|----------|
| tiny | ⭐⭐⭐⭐⭐ | ⭐⭐ | ~39 MB | Very fast, lower accuracy |
| base | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~74 MB | Good balance |
| small | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~244 MB | Better accuracy |
| medium | ⭐⭐ | ⭐⭐⭐⭐⭐ | ~769 MB | High accuracy |
| large | ⭐ | ⭐⭐⭐⭐⭐ | ~1.5 GB | Best accuracy |

## Processing Modes

### Parallel Processing
- **Pros**: Faster completion, better CPU utilization
- **Cons**: Higher memory usage, less detailed per-file progress
- **Best for**: Multiple files, powerful machines

### Sequential Processing
- **Pros**: Lower memory usage, detailed progress per file
- **Cons**: Slower overall completion
- **Best for**: Single files, limited memory

## Configuration

The application automatically creates necessary directories:
- `input_media/` - Place your source files here
- `output_transcriptions/` - Transcriptions are saved with timestamps
- `temp/` - Temporary processing files (auto-cleaned)

## Output Format

Transcriptions are saved as text files with the naming convention:
```
{original_filename}_{timestamp}.txt
```

Example: `video_20241201_143022.txt`

## Performance Tips

1. **Use Parallel Processing**: For multiple files on multi-core systems
2. **Choose Appropriate Model**: Balance speed vs accuracy based on your needs
3. **Sort by Size**: Smaller files are processed first for faster feedback
4. **Monitor Memory**: Large files with large models need significant RAM

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster processing)

## Dependencies

- `openai-whisper` - Core transcription engine
- `torch` - Machine learning framework
- `tqdm` - Progress bars
- `soundfile` - Audio file handling
- `librosa` - Audio processing

## Troubleshooting

### Common Issues

1. **"FFmpeg not found"**
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg

   # macOS
   brew install ffmpeg

   # Windows
   # Download from https://ffmpeg.org/
   ```

2. **CUDA Issues**
   - Ensure you have compatible NVIDIA drivers
   - Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

3. **Memory Errors**
   - Use smaller models (tiny/base)
   - Process files sequentially
   - Ensure adequate system RAM

### Getting Help

- Check the logs in the console output
- Ensure your files are in supported formats
- Try with a smaller test file first

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Changelog

### v1.0.0
- Initial release
- Interactive CLI interface
- Parallel and sequential processing
- Support for all Whisper models
- Comprehensive error handling

---

**Happy transcribing! 🎬📝**
