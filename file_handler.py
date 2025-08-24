"""
File handling utilities for the transcriber application
"""

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class FileHandler:
    """Handles all file operations for the transcriber."""

    def __init__(self, input_dir: Path, output_dir: Path, temp_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.temp_dir = temp_dir

        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    def get_media_files(self, supported_extensions: List[str]) -> List[Path]:
        """Get all supported media files from input directory (optimized)."""
        media_files = []

        for ext in supported_extensions:
            # Use glob for faster file discovery
            media_files.extend(self.input_dir.glob(f"*{ext}"))
            media_files.extend(self.input_dir.glob(f"*{ext.upper()}"))

        # Remove duplicates (in case of case-insensitive filesystems)
        media_files = list(set(media_files))

        # Sort by file size (smaller first for faster completion)
        media_files.sort(key=lambda x: x.stat().st_size)

        return media_files

    def get_file_info(self, file_path: Path) -> Dict:
        """Get detailed information about a file."""
        stat = file_path.stat()

        return {
            "path": file_path,
            "name": file_path.name,
            "stem": file_path.stem,
            "suffix": file_path.suffix,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "hash": self.get_file_hash(file_path),
        }

    def get_file_hash(self, file_path: Path, algorithm: str = "md5") -> str:
        """Calculate file hash for integrity checking."""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def estimate_duration(self, file_path: Path, file_type: str) -> float:
        """Estimate audio/video duration based on file size."""
        size_mb = file_path.stat().st_size / (1024 * 1024)

        if file_type == "audio":
            # Rough estimates: audio ~1MB/min
            return size_mb
        else:
            # Video ~10MB/min
            return size_mb / 10

    def generate_output_filename(
        self, input_file: Path, timestamp_format: str = "%Y%m%d_%H%M%S"
    ) -> str:
        """Generate output filename with timestamp."""
        timestamp = datetime.now().strftime(timestamp_format)
        return f"{input_file.stem}_{timestamp}.txt"

    def save_transcription(
        self, content: str, output_path: Path, metadata: Optional[Dict] = None
    ):
        """Save transcription to file with optional metadata."""
        with open(output_path, "w", encoding="utf-8") as f:
            if metadata:
                # Add metadata header
                f.write("# Transcription Metadata\n")
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")
                f.write("#" + "=" * 50 + "\n\n")

            f.write(content)

    def get_existing_transcriptions(self) -> List[Path]:
        """Get list of existing transcription files."""
        return list(self.output_dir.glob("*.txt"))

    def get_transcription_info(self, transcription_path: Path) -> Dict:
        """Get information about a transcription file."""
        stat = transcription_path.stat()

        return {
            "path": transcription_path,
            "name": transcription_path.name,
            "size_bytes": stat.st_size,
            "size_kb": stat.st_size / 1024,
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "created": datetime.fromtimestamp(stat.st_ctime),
        }

    def cleanup_temp_files(self, pattern: str = "*"):
        """Clean up temporary files."""
        try:
            for temp_file in self.temp_dir.glob(pattern):
                if temp_file.is_file():
                    temp_file.unlink()
                elif temp_file.is_dir():
                    shutil.rmtree(temp_file)
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")

    def check_disk_space(self, required_gb: float = 1.0) -> Tuple[bool, float]:
        """Check if there's enough disk space available."""
        try:
            stat = os.statvfs(self.output_dir)
            available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            return available_gb >= required_gb, available_gb
        except Exception:
            return True, 0.0  # Assume enough space if we can't check

    def validate_file(
        self, file_path: Path, max_size_gb: float = 2.0
    ) -> Tuple[bool, str]:
        """Validate a media file."""
        if not file_path.exists():
            return False, "File does not exist"

        if not file_path.is_file():
            return False, "Path is not a file"

        if not os.access(file_path, os.R_OK):
            return False, "File is not readable"

        file_size_gb = file_path.stat().st_size / (1024**3)
        if file_size_gb > max_size_gb:
            return False, f"File too large: {file_size_gb:.2f} GB (max: {max_size_gb:.2f} GB)"

        return True, "OK"

    def create_backup(
        self, file_path: Path, backup_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Create a backup of a file."""
        if not backup_dir:
            backup_dir = self.temp_dir / "backups"
            backup_dir.mkdir(exist_ok=True)

        backup_path = backup_dir / f"{file_path.name}.backup"
        try:
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
            return None

    def get_processing_stats(self) -> Dict:
        """Get statistics about processed files."""
        transcriptions = self.get_existing_transcriptions()

        total_size = sum(t.stat().st_size for t in transcriptions)
        total_files = len(transcriptions)

        if transcriptions:
            oldest = min(t.stat().st_mtime for t in transcriptions)
            newest = max(t.stat().st_mtime for t in transcriptions)
        else:
            oldest = newest = None

        return {
            "total_transcriptions": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024**2),
            "oldest_transcription": oldest,
            "newest_transcription": newest,
            "avg_size_bytes": total_size / max(total_files, 1),
        }
