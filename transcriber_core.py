#!/usr/bin/env python3
"""
Consolidated Transcriber Core Module

This is the main transcriber functionality that combines:
- Fast processing from the original transcriber.py
- Enhanced error handling and features from transcriber_enhanced.py
- Clean, unified interface for both interactive and daemon modes
"""

import gc
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised in dependency-light test envs
    def tqdm(iterable, **_kwargs):
        return iterable

from config import TranscriberConfig, get_config
from file_handler import FileHandler
from logger import get_logger
from media_pipeline import (
    PreparedAudio,
    MediaProcessingError,
    probe_media,
    prepare_audio_for_transcription,
    merge_chunk_results,
)
from subtitle_formats import segments_to_srt, segments_to_vtt

try:
    import torch
except ImportError:  # pragma: no cover - exercised in dependency-light test envs
    torch = None

try:
    import whisper
except ImportError:  # pragma: no cover - exercised in dependency-light test envs
    whisper = None


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
class ProcessingResult:
    """Result of processing a single file."""
    success: bool
    file_path: Path
    output_path: Optional[Path]
    processing_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingStats:
    """Statistics for a batch processing session."""
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    total_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class TranscriberCore:
    """
    Unified transcriber core with enhanced features.

    Features:
    - Efficient model loading and reuse
    - Enhanced error handling with retry logic
    - GPU support detection and optimization
    - Memory optimization and garbage collection
    - Detailed progress tracking
    - Subtitle generation support
    - Batch processing with multiprocessing
    """

    def __init__(self, config: Optional[TranscriberConfig] = None):
        self.config = config or get_config()
        self.logger = get_logger("transcriber")

        # Core components
        self.file_handler = FileHandler(
            self.config.input_dir, self.config.output_dir, self.config.temp_dir
        )

        # Model management
        self.model: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.model_load_time: Optional[float] = None

        # Processing state
        self.is_processing = False
        self.current_file: Optional[Path] = None

        # Hardware detection
        self._detect_hardware()

    def _detect_hardware(self):
        """Detect available hardware for optimization."""
        try:
            if torch is None:
                self.logger.warning("⚠️  PyTorch not installed, assuming CPU-only execution")
                self.has_cuda = False
                return

            self.has_cuda = torch.cuda.is_available()
            if self.has_cuda:
                self.gpu_count = torch.cuda.device_count()
                self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_count > 0 else "Unknown"
                self.logger.info(f"✅ CUDA available: {self.gpu_count} GPU(s) - {self.gpu_name}")
            else:
                self.logger.info("⚠️  CUDA not available, using CPU")
        except Exception as e:
            self.logger.warning(f"Hardware detection failed: {e}")
            self.has_cuda = False

    def get_media_files(self) -> List[Path]:
        """Get all supported media files from input directory."""
        supported_extensions = self.config.get_supported_extensions()
        return self.file_handler.get_media_files(supported_extensions)

    def load_model(self, model_name: str) -> Any:
        """
        Load Whisper model with enhanced error handling.

        Args:
            model_name: Name of the Whisper model to load

        Returns:
            Loaded Whisper model

        Raises:
            ModelLoadError: If model loading fails
        """
        if self.model is None or self.model_name != model_name:
            start_time = time.time()

            try:
                if whisper is None:
                    raise ModelLoadError(
                        "openai-whisper is not installed. Install dependencies before transcribing."
                    )

                self.logger.info(f"📥 Loading {model_name} model...")

                # Set device based on availability
                device = "cuda" if self.has_cuda else "cpu"

                self.model = whisper.load_model(model_name, device=device)
                self.model_name = model_name
                self.model_load_time = time.time() - start_time

                self.logger.info(f"Model loaded in {self.model_load_time:.2f}s")
                gc.collect()

            except Exception as e:
                error_msg = f"Failed to load model {model_name}: {e}"
                self.logger.error(error_msg)
                raise ModelLoadError(error_msg)

        return self.model

    def transcribe_file(
        self,
        file_path: Path,
        model_name: str,
        retry_count: int = 3
    ) -> ProcessingResult:
        """
        Transcribe a single file with enhanced error handling.

        Args:
            file_path: Path to the file to transcribe
            model_name: Whisper model to use
            retry_count: Number of retry attempts on failure

        Returns:
            ProcessingResult with transcription details
        """
        start_time = time.time()
        self.current_file = file_path
        self.is_processing = True
        prepared_audio: Optional[PreparedAudio] = None

        try:
            self.logger.info(f"🎬 Transcribing: {file_path.name}")

            # Validate file
            is_valid, error_msg = self.file_handler.validate_file(file_path)
            if not is_valid:
                raise TranscriptionError(f"File validation failed: {error_msg}")

            probe_info = probe_media(file_path)
            is_valid_probe, probe_error = self.file_handler.validate_probe_result(
                probe_info.duration_seconds, probe_info.codec_name
            )
            if not is_valid_probe:
                raise TranscriptionError(f"Media validation failed: {probe_error}")

            self.logger.info(
                "   🎧 Detected audio: "
                f"container={probe_info.container}, codec={probe_info.codec_name}, "
                f"duration={probe_info.duration_seconds / 60:.1f} min, "
                f"sample_rate={probe_info.sample_rate_hz or 'unknown'} Hz, "
                f"channels={probe_info.channels or 'unknown'}"
            )

            prepared_audio = prepare_audio_for_transcription(
                file_path=file_path,
                temp_dir=self.config.temp_dir,
                probe_info=probe_info,
                preprocess_audio=self.config.preprocess_audio,
                long_audio_threshold_minutes=self.config.long_audio_threshold_minutes,
                chunk_duration_minutes=self.config.chunk_duration_minutes,
                chunk_overlap_seconds=self.config.chunk_overlap_seconds,
            )
            self.logger.info(
                f"   🧹 Prepared audio at 16 kHz mono: {prepared_audio.normalized_path.name}"
            )
            if len(prepared_audio.chunks) > 1:
                self.logger.info(
                    "   ✂️  Auto-chunked into "
                    f"{len(prepared_audio.chunks)} pieces "
                    f"({self.config.chunk_duration_minutes} min with "
                    f"{self.config.chunk_overlap_seconds}s overlap)"
                )

            # Load model after decode/probe succeeds.
            model = self.load_model(model_name)

            result = self._transcribe_prepared_audio(model, prepared_audio, retry_count)

            if not result or "text" not in result:
                raise TranscriptionError("Transcription failed - no text returned")

            transcription = result["text"].strip()
            processing_time = time.time() - start_time

            output_format = (self.config.output_format or "txt").lower()
            # Generate output filename and path
            output_filename = self.file_handler.generate_output_filename(
                file_path, self.config.timestamp_format, output_format
            )
            output_path = self.config.output_dir / output_filename

            # Create metadata
            metadata = None
            if self.config.include_metadata:
                metadata = {
                    "source_file": file_path.name,
                    "model": model_name,
                    "processing_time_seconds": processing_time,
                    "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                    "timestamp": datetime.now().isoformat(),
                    "device": "cuda" if self.has_cuda else "cpu",
                    "model_load_time": self.model_load_time,
                    "duration_seconds": probe_info.duration_seconds,
                    "audio_codec": probe_info.codec_name,
                    "audio_container": probe_info.container,
                    "sample_rate_hz": probe_info.sample_rate_hz,
                    "channels": probe_info.channels,
                    "chunk_count": len(prepared_audio.chunks),
                }

            formatted_output, allow_metadata = self._format_output(result)
            if not allow_metadata:
                metadata = None

            # Save transcription
            self.file_handler.save_transcription(formatted_output, output_path, metadata)

            self.logger.info(f"   ✅ Completed in {processing_time:.1f} seconds")
            self.logger.info(f"   📝 Characters: {len(transcription):,}")
            self.logger.info(f"   📄 Saved: {output_filename}")

            return ProcessingResult(
                success=True,
                file_path=file_path,
                output_path=output_path,
                processing_time=processing_time,
                metadata=metadata
            )

        except MediaProcessingError as e:
            processing_time = time.time() - start_time
            error_msg = f"Error preparing {file_path.name}: {e}"
            self.logger.error(f"   ❌ {error_msg}")

            return ProcessingResult(
                success=False,
                file_path=file_path,
                output_path=None,
                processing_time=processing_time,
                error_message=error_msg
            )
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {file_path.name}: {e}"
            self.logger.error(f"   ❌ {error_msg}")

            return ProcessingResult(
                success=False,
                file_path=file_path,
                output_path=None,
                processing_time=processing_time,
                error_message=error_msg
            )

        finally:
            self.is_processing = False
            self.current_file = None
            if prepared_audio is not None:
                prepared_audio.cleanup()
            gc.collect()

    def _format_output(self, result: Dict[str, Any]) -> Tuple[str, bool]:
        """Format transcription output based on config."""
        output_format = (self.config.output_format or "txt").lower()
        transcription = (result.get("text") or "").strip()
        if output_format == "txt":
            return transcription, True

        segments = result.get("segments") or []
        if not segments:
            self.logger.warning("No segments available; falling back to plain text output.")
            return transcription, True

        if output_format == "srt":
            return segments_to_srt(segments), False
        if output_format == "vtt":
            return segments_to_vtt(segments), False

        self.logger.warning(f"Unknown output format '{output_format}', using txt.")
        return transcription, True

    def _transcribe_with_retry(self, model, file_path: Path, max_retries: int) -> Optional[Dict]:
        """Transcribe with retry logic for handling temporary failures."""
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"   🎵 Processing audio (attempt {attempt + 1}/{max_retries})...")
                result = model.transcribe(str(file_path), verbose=False)

                if result and "text" in result:
                    return result
                else:
                    raise TranscriptionError("Empty transcription result")

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    self.logger.warning(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
                    self.logger.info(f"   ⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise e

        return None

    def _transcribe_prepared_audio(
        self,
        model: Any,
        prepared_audio: PreparedAudio,
        retry_count: int,
    ) -> Dict[str, Any]:
        """Transcribe prepared audio and merge chunked results when needed."""
        if len(prepared_audio.chunks) == 1:
            return self._transcribe_with_retry(
                model, prepared_audio.chunks[0].path, retry_count
            ) or {"text": "", "segments": []}

        chunk_results = []
        for index, chunk in enumerate(prepared_audio.chunks, start=1):
            self.logger.info(
                f"   🧩 Chunk {index}/{len(prepared_audio.chunks)} "
                f"({chunk.start_seconds / 60:.1f}-{chunk.end_seconds / 60:.1f} min)"
            )
            result = self._transcribe_with_retry(model, chunk.path, retry_count)
            if not result:
                raise TranscriptionError(
                    f"Chunk {index} returned no transcription result"
                )
            chunk_results.append((chunk, result))

        self.logger.info("   🔗 Merging chunk transcripts into one timeline")
        return merge_chunk_results(chunk_results)

    def process_files_sequential(
        self,
        files: List[Path],
        model_name: str
    ) -> Tuple[int, List[str]]:
        """Process files sequentially with progress tracking."""
        if not files:
            return 0, []

        total_files = len(files)
        successful = 0
        failed = []

        self.logger.info(f"🎯 Processing {total_files} file(s) with {model_name} model")
        self.logger.info("=" * 60)

        for i, file_path in enumerate(files, 1):
            self.logger.info(f"\n📁 File {i}/{total_files}")

            result = self.transcribe_file(file_path, model_name)

            if result.success:
                successful += 1
            else:
                failed.append(file_path.name)

            # Progress update
            progress = (i / total_files) * 100
            self.logger.info(f"📊 Progress: {progress:.0f}% ({i}/{total_files})")

        # Summary
        self.logger.info(f"\n🎉 Completed: {successful}/{total_files} files")
        if failed:
            self.logger.warning(f"❌ Failed: {', '.join(failed)}")

        return successful, failed

    def process_files_parallel(
        self,
        files: List[Path],
        model_name: str,
        max_workers: Optional[int] = None
    ) -> Tuple[int, List[str]]:
        """Process multiple files in parallel."""
        if not files:
            return 0, []

        if max_workers is None:
            max_workers = self.config.get_auto_workers()

        self.logger.info(f"🚀 Processing {len(files)} file(s) with {model_name} model")
        self.logger.info(f"   Using {max_workers} parallel workers")
        self.logger.info("=" * 60)

        # Prepare arguments for multiprocessing
        config_data = asdict(self.config)
        args = [(file_path, model_name, config_data) for file_path in files]

        successful = 0
        failed = []

        # Process files in parallel with progress bar
        with mp.Pool(processes=max_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(self._worker_transcribe, args),
                    total=len(files),
                    desc="Transcribing files",
                    unit="file",
                    disable=not self.config.progress_bar,
                )
            )

        # Process results
        for success, filename, output_path in results:
            if success:
                successful += 1
                self.logger.info(f"✅ {filename} completed")
            else:
                failed.append(filename)
                self.logger.error(f"❌ {filename} failed: {output_path}")

        # Summary
        self.logger.info(f"\n🎉 Completed: {successful}/{len(files)} files")
        if failed:
            self.logger.warning(f"❌ Failed: {', '.join(failed)}")

        return successful, failed

    @staticmethod
    def _worker_transcribe(
        args: Tuple[Path, str, Dict[str, Any]]
    ) -> Tuple[bool, str, Optional[str]]:
        """Worker function for multiprocessing."""
        file_path, model_name, config_data = args

        try:
            worker = TranscriberCore(TranscriberConfig(**config_data))
            result = worker.transcribe_file(file_path, model_name)
            if result.success and result.output_path:
                return True, file_path.name, str(result.output_path)
            return False, file_path.name, result.error_message
        except Exception as e:
            return False, file_path.name, str(e)

    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        return {
            "is_processing": self.is_processing,
            "current_file": str(self.current_file) if self.current_file else None,
            "model_loaded": self.model_name is not None,
            "model_name": self.model_name,
            "has_cuda": self.has_cuda,
        }

    def show_files(self) -> List[Path]:
        """Display available media files."""
        files = self.get_media_files()

        if not files:
            self.logger.warning("❌ No media files found in input_media/")
            self.logger.info("\nSupported formats: mp4, avi, mov, mkv, webm, flv, mp3, wav, m4a, aac, flac, ogg")
            return []

        self.logger.info(f"🎬 Found {len(files)} media file(s):")
        for i, file_path in enumerate(files, 1):
            file_info = self.file_handler.get_file_info(file_path)
            self.logger.info(f"{i:2d}. {file_path.name} ({file_info['size_mb']:.1f} MB)")

        return files

    def show_outputs(self) -> None:
        """Display existing transcription files."""
        output_files = self.file_handler.get_existing_transcriptions(["txt", "srt", "vtt"])

        if not output_files:
            self.logger.info("📁 No transcription files found")
            return

        self.logger.info(f"\n📁 Found {len(output_files)} transcription file(s):")
        for file_path in sorted(
            output_files, key=lambda x: x.stat().st_mtime, reverse=True
        ):
            transcription_info = self.file_handler.get_transcription_info(file_path)
            self.logger.info(
                f"   {file_path.name} ({transcription_info['size_kb']:.1f} KB, {transcription_info['modified'].strftime('%Y-%m-%d %H:%M')})"
            )
