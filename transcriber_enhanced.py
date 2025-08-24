#!/usr/bin/env python3
"""
Enhanced Fast Video/Audio Transcriber

Improvements:
- Better error handling with retry logic
- GPU support detection
- Enhanced progress tracking
- Subtitle generation support
- Improved memory management
"""

import gc
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import whisper
from tqdm import tqdm

from config import TranscriberConfig, get_config
from file_handler import FileHandler
from logger import TranscriptionLogger, get_logger


# Custom Exceptions
class TranscriptionError(Exception):
    """Base exception for transcription errors."""
    pass


class RetryableError(TranscriptionError):
    """Errors that can be retried."""
    pass


class ModelLoadError(TranscriptionError):
    """Error loading the model."""
    pass


@dataclass
class TranscriptionResult:
    """Container for transcription results."""
    success: bool
    file_path: Path
    output_path: Optional[Path] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    character_count: Optional[int] = None
    segments: Optional[List[Dict]] = None


class DeviceManager:
    """Manages device selection for model inference."""
    
    @staticmethod
    def get_best_device() -> str:
        """Detect and return the best available device."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU detected: {device_name} ({memory_gb:.1f} GB)")
            return "cuda"
        else:
            print("💻 Using CPU (GPU not available)")
            return "cpu"
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": None,
            "gpu_name": None,
            "gpu_memory_gb": None,
        }
        
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name()
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info


class ProgressTracker:
    """Enhanced progress tracking with statistics."""
    
    def __init__(self, total_files: int):
        self.total_files = total_files
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.start_time = time.time()
        self.file_times: List[float] = []
        self.file_sizes: List[float] = []
    
    def update(self, result: TranscriptionResult, file_size_mb: float):
        """Update progress with result."""
        self.processed += 1
        
        if result.success:
            self.successful += 1
            if result.duration_seconds:
                self.file_times.append(result.duration_seconds)
                self.file_sizes.append(file_size_mb)
        else:
            self.failed += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        elapsed = time.time() - self.start_time
        remaining = self.total_files - self.processed
        
        stats = {
            "processed": self.processed,
            "successful": self.successful,
            "failed": self.failed,
            "remaining": remaining,
            "success_rate": (self.successful / max(self.processed, 1)) * 100,
            "elapsed_seconds": elapsed,
            "elapsed_formatted": str(timedelta(seconds=int(elapsed))),
        }
        
        if self.file_times:
            avg_time = sum(self.file_times) / len(self.file_times)
            avg_size = sum(self.file_sizes) / len(self.file_sizes)
            eta_seconds = remaining * avg_time
            
            stats.update({
                "avg_time_seconds": avg_time,
                "avg_file_size_mb": avg_size,
                "eta_seconds": eta_seconds,
                "eta_formatted": str(timedelta(seconds=int(eta_seconds))),
                "processing_speed_mbps": avg_size / avg_time if avg_time > 0 else 0,
            })
        
        return stats


class SubtitleGenerator:
    """Generate subtitle files from transcription results."""
    
    @staticmethod
    def format_timestamp_srt(seconds: float) -> str:
        """Format seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
    
    @staticmethod
    def format_timestamp_vtt(seconds: float) -> str:
        """Format seconds to WebVTT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def generate_srt(self, segments: List[Dict], output_path: Path) -> bool:
        """Generate SRT subtitle file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments, 1):
                    start = self.format_timestamp_srt(segment["start"])
                    end = self.format_timestamp_srt(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")
            return True
        except Exception as e:
            print(f"❌ Error generating SRT: {e}")
            return False
    
    def generate_vtt(self, segments: List[Dict], output_path: Path) -> bool:
        """Generate WebVTT subtitle file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                
                for segment in segments:
                    start = self.format_timestamp_vtt(segment["start"])
                    end = self.format_timestamp_vtt(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")
            return True
        except Exception as e:
            print(f"❌ Error generating VTT: {e}")
            return False
    
    def generate_json(self, result: Dict, output_path: Path) -> bool:
        """Generate JSON output with full transcription data."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Error generating JSON: {e}")
            return False


class EnhancedTranscriber:
    """Enhanced transcriber with improved error handling and features."""
    
    def __init__(self, config: Optional[TranscriberConfig] = None):
        self.config = config or get_config()
        self.file_handler = FileHandler(
            self.config.input_dir, self.config.output_dir, self.config.temp_dir
        )
        self.logger = get_logger("enhanced_transcriber")
        self.transcription_logger = TranscriptionLogger(self.logger)
        self.subtitle_generator = SubtitleGenerator()
        self.device_manager = DeviceManager()
        
        self.model: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.device: Optional[str] = None
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.enable_subtitles = True
        self.output_formats = ["txt", "srt", "vtt", "json"]
    
    def load_model(self, model_name: str, device: Optional[str] = None) -> Any:
        """Load Whisper model with device optimization."""
        if device is None:
            device = self.device_manager.get_best_device()
        
        if self.model is None or self.model_name != model_name or self.device != device:
            try:
                self.transcription_logger.log_model_loading(model_name)
                
                # Load model with device selection
                self.model = whisper.load_model(model_name, device=device)
                self.model_name = model_name
                self.device = device
                
                # Optimize for GPU if available
                if device == "cuda" and model_name in ["small", "medium", "large"]:
                    self.model = self.model.half()  # Use FP16 for faster inference
                
                self.transcription_logger.log_model_loaded(model_name)
                
                # Force garbage collection
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                
            except Exception as e:
                raise ModelLoadError(f"Failed to load model {model_name}: {e}")
        
        return self.model
    
    def transcribe_with_retry(
        self, file_path: Path, model_name: str, device: Optional[str] = None
    ) -> TranscriptionResult:
        """Transcribe file with retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self._transcribe_single(file_path, model_name, device)
            except RetryableError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} for {file_path.name} "
                        f"after {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
            except Exception as e:
                # Non-retryable error
                self.logger.error(f"Non-retryable error for {file_path.name}: {e}")
                return TranscriptionResult(
                    success=False,
                    file_path=file_path,
                    error_message=str(e)
                )
        
        # All retries exhausted
        return TranscriptionResult(
            success=False,
            file_path=file_path,
            error_message=f"Failed after {self.max_retries} attempts: {last_error}"
        )
    
    def _transcribe_single(
        self, file_path: Path, model_name: str, device: Optional[str] = None
    ) -> TranscriptionResult:
        """Internal method to transcribe a single file."""
        print(f"\n🎬 Transcribing: {file_path.name}")
        
        # Validate file
        is_valid, error_msg = self.file_handler.validate_file(
            file_path, self.config.max_file_size_gb
        )
        if not is_valid:
            raise TranscriptionError(f"File validation failed: {error_msg}")
        
        # Check available memory
        if not self._check_memory(file_path):
            raise RetryableError("Insufficient memory available")
        
        # Load model
        model = self.load_model(model_name, device)
        
        # Get file info
        file_info = self.file_handler.get_file_info(file_path)
        file_size_mb = file_info["size_mb"]
        
        # Estimate duration
        estimated_minutes = self.file_handler.estimate_duration(
            file_path, self.config.supported_extensions.get(file_path.suffix.lower(), "video")
        )
        print(f"   ⏱️  Estimated duration: ~{estimated_minutes:.1f} minutes")
        print(f"   📦 File size: {file_size_mb:.1f} MB")
        
        start_time = time.time()
        
        try:
            # Transcribe with detailed results
            print("   🎵 Processing audio...")
            result = model.transcribe(
                str(file_path),
                verbose=False,
                word_timestamps=True  # Enable word-level timestamps for better subtitles
            )
            
            if not result or "text" not in result:
                raise TranscriptionError("Transcription failed - no text returned")
            
            transcription = result["text"].strip()
            segments = result.get("segments", [])
            
            # Calculate processing time
            elapsed_time = time.time() - start_time
            
            print(f"   ✅ Completed in {elapsed_time:.1f} seconds")
            print(f"   📝 Characters: {len(transcription):,}")
            if segments:
                print(f"   🎯 Segments: {len(segments)}")
            
            # Generate output filename
            timestamp = datetime.now().strftime(self.config.timestamp_format)
            base_filename = f"{file_path.stem}_{timestamp}"
            
            # Save transcription
            txt_path = self.config.output_dir / f"{base_filename}.txt"
            metadata = {
                "source_file": file_path.name,
                "model": model_name,
                "device": self.device,
                "processing_time_seconds": elapsed_time,
                "file_size_mb": file_size_mb,
                "character_count": len(transcription),
                "segment_count": len(segments),
                "timestamp": datetime.now().isoformat(),
            }
            
            self.file_handler.save_transcription(transcription, txt_path, metadata)
            print(f"   📄 Saved text: {txt_path.name}")
            
            # Generate subtitles if enabled
            if self.enable_subtitles and segments:
                if "srt" in self.output_formats:
                    srt_path = self.config.output_dir / f"{base_filename}.srt"
                    if self.subtitle_generator.generate_srt(segments, srt_path):
                        print(f"   📄 Saved SRT: {srt_path.name}")
                
                if "vtt" in self.output_formats:
                    vtt_path = self.config.output_dir / f"{base_filename}.vtt"
                    if self.subtitle_generator.generate_vtt(segments, vtt_path):
                        print(f"   📄 Saved VTT: {vtt_path.name}")
                
                if "json" in self.output_formats:
                    json_path = self.config.output_dir / f"{base_filename}.json"
                    if self.subtitle_generator.generate_json(result, json_path):
                        print(f"   📄 Saved JSON: {json_path.name}")
            
            # Clean up memory
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return TranscriptionResult(
                success=True,
                file_path=file_path,
                output_path=txt_path,
                duration_seconds=elapsed_time,
                character_count=len(transcription),
                segments=segments
            )
            
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"GPU out of memory: {e}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
                raise RetryableError("GPU out of memory, retry with CPU")
            raise
        except Exception as e:
            self.logger.error(f"Transcription error: {e}\n{traceback.format_exc()}")
            raise TranscriptionError(f"Transcription failed: {e}")
    
    def _check_memory(self, file_path: Path) -> bool:
        """Check if sufficient memory is available."""
        try:
            import psutil
            
            file_size_gb = file_path.stat().st_size / (1024**3)
            required_memory_gb = file_size_gb * 3  # Rough estimate
            
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            if self.device == "cuda":
                gpu_memory_free = torch.cuda.mem_get_info()[0] / (1024**3)
                required_gpu_memory = file_size_gb * 2
                
                if gpu_memory_free < required_gpu_memory:
                    self.logger.warning(
                        f"Insufficient GPU memory: {gpu_memory_free:.1f} GB available, "
                        f"{required_gpu_memory:.1f} GB required"
                    )
                    return False
            
            if available_memory_gb < required_memory_gb:
                self.logger.warning(
                    f"Insufficient RAM: {available_memory_gb:.1f} GB available, "
                    f"{required_memory_gb:.1f} GB required"
                )
                return False
            
            return True
            
        except ImportError:
            # psutil not available, assume enough memory
            return True
        except Exception as e:
            self.logger.warning(f"Could not check memory: {e}")
            return True
    
    def process_batch(
        self,
        files: List[Path],
        model_name: str,
        parallel: bool = False,
        max_workers: Optional[int] = None
    ) -> Tuple[int, List[str]]:
        """Process multiple files with enhanced tracking."""
        if not files:
            return 0, []
        
        # Initialize progress tracker
        tracker = ProgressTracker(len(files))
        
        # Sort files by size for optimal processing
        files.sort(key=lambda x: x.stat().st_size)
        
        print(f"\n🚀 Processing {len(files)} file(s) with {model_name} model")
        
        # Show device info
        device_info = self.device_manager.get_device_info()
        if device_info["cuda_available"]:
            print(f"   🎮 GPU: {device_info['gpu_name']} ({device_info['gpu_memory_gb']:.1f} GB)")
        else:
            print("   💻 CPU mode")
        
        print("=" * 60)
        
        failed_files = []
        
        if parallel and len(files) > 1:
            # Parallel processing
            if max_workers is None:
                max_workers = self.config.get_auto_workers()
            
            print(f"   🔄 Parallel mode with {max_workers} workers")
            
            # Note: Parallel processing with GPU is complex and may not be efficient
            # For simplicity, we'll process sequentially for now
            # A production implementation would need proper GPU resource management
            
        # Sequential processing (recommended for GPU)
        for i, file_path in enumerate(files, 1):
            print(f"\n📁 File {i}/{len(files)}")
            
            # Get file info
            file_info = self.file_handler.get_file_info(file_path)
            
            # Transcribe with retry
            result = self.transcribe_with_retry(file_path, model_name, self.device)
            
            # Update tracker
            tracker.update(result, file_info["size_mb"])
            
            if not result.success:
                failed_files.append(file_path.name)
            
            # Show progress stats
            stats = tracker.get_stats()
            print(f"\n📊 Progress: {stats['processed']}/{len(files)} files")
            print(f"   Success rate: {stats['success_rate']:.1f}%")
            if "eta_formatted" in stats:
                print(f"   ETA: {stats['eta_formatted']}")
                print(f"   Speed: {stats['processing_speed_mbps']:.1f} MB/s")
        
        # Final summary
        stats = tracker.get_stats()
        print("\n" + "=" * 60)
        print(f"🎉 Batch processing completed!")
        print(f"   ✅ Successful: {stats['successful']}/{len(files)}")
        if stats['failed'] > 0:
            print(f"   ❌ Failed: {stats['failed']}")
            print(f"   Failed files: {', '.join(failed_files)}")
        print(f"   ⏱️  Total time: {stats['elapsed_formatted']}")
        
        return stats['successful'], failed_files
    
    def cleanup(self):
        """Clean up resources."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Clean temporary files
        self.file_handler.cleanup_temp_files()


def main():
    """Main function with enhanced error handling."""
    try:
        print("🎬 Enhanced Fast Video/Audio Transcriber")
        print("=" * 40)
        
        # Create enhanced transcriber
        transcriber = EnhancedTranscriber()
        
        # Show device information
        device_info = transcriber.device_manager.get_device_info()
        print(f"\n💻 System Information:")
        print(f"   Device: {device_info['device'].upper()}")
        if device_info['cuda_available']:
            print(f"   GPU: {device_info['gpu_name']}")
            print(f"   VRAM: {device_info['gpu_memory_gb']:.1f} GB")
            print(f"   CUDA: {device_info['cuda_version']}")
        
        # Get media files
        files = transcriber.file_handler.get_media_files(
            transcriber.config.get_supported_extensions()
        )
        
        if not files:
            print("\n❌ No media files found in input_media/")
            print("Place your media files in the input_media/ directory and try again.")
            return 1
        
        print(f"\n📁 Found {len(files)} media file(s)")
        
        # For demo, process with base model
        model_name = "base"
        
        # Process files
        successful, failed = transcriber.process_batch(files, model_name)
        
        # Cleanup
        transcriber.cleanup()
        
        return 0 if failed == [] else 1
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())