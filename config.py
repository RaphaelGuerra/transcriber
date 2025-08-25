"""
Configuration management for Fast Transcriber
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class TranscriberConfig:
    """Configuration class for the transcriber application."""

    # Directory paths
    input_dir: Path = field(default_factory=lambda: Path("input_media"))
    output_dir: Path = field(default_factory=lambda: Path("output_transcriptions"))
    temp_dir: Path = field(default_factory=lambda: Path("temp"))

    # Model settings
    default_model: str = "base"
    available_models: List[str] = field(
        default_factory=lambda: ["tiny", "base", "small", "medium", "large"]
    )

    # Processing settings
    max_workers: Optional[int] = None  # None = auto-detect
    parallel_processing: bool = True
    memory_limit_gb: float = 4.0

    # File settings
    supported_extensions: Dict[str, str] = field(
        default_factory=lambda: {
            # Video formats
            ".mp4": "video",
            ".avi": "video",
            ".mov": "video",
            ".mkv": "video",
            ".webm": "video",
            ".flv": "video",
            # Audio formats
            ".mp3": "audio",
            ".wav": "audio",
            ".m4a": "audio",
            ".aac": "audio",
            ".flac": "audio",
            ".ogg": "audio",
        }
    )

    # Output settings
    output_format: str = "txt"
    include_metadata: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"

    # Performance settings
    chunk_size_mb: int = 25
    max_file_size_gb: float = 2.0

    # UI settings
    verbose: bool = True
    progress_bar: bool = True

    # Daemon settings
    daemon_mode: bool = False
    scan_interval_seconds: int = 30
    sleep_monitoring_enabled: bool = True
    auto_resume_after_wake: bool = True
    max_concurrent_files: int = 1  # For daemon mode, process files sequentially
    daemon_log_file: Optional[Path] = None
    pid_file_path: Optional[Path] = None

    def __post_init__(self):
        """Initialize configuration and create directories."""
        # Convert string paths to Path objects if needed
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.temp_dir, str):
            self.temp_dir = Path(self.temp_dir)

        # Create directories
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "TranscriberConfig":
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            return cls()  # Return default config if file doesn't exist

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert string paths back to Path objects
        for key in ["input_dir", "output_dir", "temp_dir"]:
            if key in data and data[key]:
                data[key] = Path(data[key])

        return cls(**data)

    def save_to_file(self, config_path: Union[str, Path]):
        """Save configuration to JSON file."""
        config_path = Path(config_path)

        # Convert Path objects to strings for JSON serialization
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, set):
                data[key] = list(value)
            else:
                data[key] = value

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_supported_extensions(self, media_type: Optional[str] = None) -> List[str]:
        """Get supported file extensions, optionally filtered by media type."""
        if media_type:
            return [
                ext
                for ext, mtype in self.supported_extensions.items()
                if mtype == media_type
            ]
        return list(self.supported_extensions.keys())

    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """Check if a file extension is supported."""
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions

    def get_auto_workers(self) -> int:
        """Get automatically determined number of workers based on system."""
        import multiprocessing as mp
        import os

        # Base workers on CPU cores
        cpu_count = mp.cpu_count()

        # Adjust based on available memory
        try:
            import psutil

            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            memory_based_workers = max(1, int(available_memory / self.memory_limit_gb))
            return min(cpu_count, memory_based_workers)
        except ImportError:
            return max(1, cpu_count - 1)  # Leave one core free

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check model validity
        if self.default_model not in self.available_models:
            issues.append(
                f"Invalid default model: {self.default_model}. Available: {self.available_models}"
            )

        # Check directory accessibility
        for dir_path, dir_name in [
            (self.input_dir, "input"),
            (self.output_dir, "output"),
            (self.temp_dir, "temp"),
        ]:
            if not dir_path.exists():
                issues.append(
                    f"{dir_name.capitalize()} directory does not exist: {dir_path}"
                )
            elif not os.access(dir_path, os.W_OK):
                issues.append(
                    f"{dir_name.capitalize()} directory is not writable: {dir_path}"
                )

        # Check file size limits
        if self.max_file_size_gb <= 0:
            issues.append("max_file_size_gb must be positive")
        if self.chunk_size_mb <= 0:
            issues.append("chunk_size_mb must be positive")

        return issues


# Default configuration instance
default_config = TranscriberConfig()


def get_config(config_path: Optional[Union[str, Path]] = None) -> TranscriberConfig:
    """Get configuration, loading from file if specified."""
    if config_path and Path(config_path).exists():
        return TranscriberConfig.from_file(config_path)
    return default_config


def save_default_config(config_path: Union[str, Path]):
    """Save default configuration to file."""
    default_config.save_to_file(config_path)
