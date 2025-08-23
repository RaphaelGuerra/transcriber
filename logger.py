"""
Logging configuration for the transcriber application
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Add color to the level name
        if hasattr(record, "levelname") and record.levelname in self.COLORS:
            colored_levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
            record.levelname = colored_levelname

        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
    color: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
        color: Whether to use colored output for console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("transcriber")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        if color:
            console_formatter = ColoredFormatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )
        else:
            console_formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
            )

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "transcriber") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(f"transcriber.{name}")


# Default logger instance
default_logger = get_logger()


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {e}")
                raise

        return wrapper

    return decorator


class ProgressLogger:
    """Helper class for logging progress information."""

    def __init__(
        self, logger: logging.Logger, total: int, description: str = "Processing"
    ):
        self.logger = logger
        self.total = total
        self.description = description
        self.current = 0

    def update(self, increment: int = 1, message: str = ""):
        """Update progress."""
        self.current += increment
        progress = (self.current / self.total) * 100

        log_message = (
            f"{self.description}: {self.current}/{self.total} ({progress:.1f}%)"
        )
        if message:
            log_message += f" - {message}"

        self.logger.info(log_message)

    def complete(self, message: str = "Completed"):
        """Mark progress as complete."""
        self.logger.info(f"{self.description}: {message}")


class TranscriptionLogger:
    """Specialized logger for transcription operations."""

    EMOJI_MAP = {
        "start": "🎬",
        "processing": "🎵",
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️",
        "file": "📄",
        "time": "⏱️",
        "progress": "📊",
    }

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_file_start(self, file_path: str, estimated_duration: float = 0):
        """Log the start of file processing."""
        duration_text = (
            f" (est. {estimated_duration:.1f} min)" if estimated_duration > 0 else ""
        )
        self.logger.info(
            f"{self.EMOJI_MAP['start']} Processing: {Path(file_path).name}{duration_text}"
        )

    def log_file_progress(self, message: str, emoji: str = "processing"):
        """Log file processing progress."""
        emoji_char = self.EMOJI_MAP.get(emoji, self.EMOJI_MAP["info"])
        self.logger.info(f"   {emoji_char} {message}")

    def log_file_success(self, file_path: str, duration: float, characters: int):
        """Log successful file completion."""
        self.logger.info(f"   {self.EMOJI_MAP['success']} Completed in {duration:.1f}s")
        self.logger.info(f"   {self.EMOJI_MAP['file']} Characters: {characters:,}")

    def log_file_error(self, file_path: str, error: str):
        """Log file processing error."""
        self.logger.error(
            f"   {self.EMOJI_MAP['error']} {Path(file_path).name}: {error}"
        )

    def log_batch_start(self, total_files: int, model: str, mode: str):
        """Log the start of batch processing."""
        mode_text = "parallel" if mode == "parallel" else "sequential"
        self.logger.info(
            f"{self.EMOJI_MAP['start']} Processing {total_files} file(s) with {model} model ({mode_text})"
        )

    def log_batch_progress(
        self, current: int, total: int, filename: str, success: bool
    ):
        """Log batch processing progress."""
        progress = (current / total) * 100
        status = self.EMOJI_MAP["success"] if success else self.EMOJI_MAP["error"]
        self.logger.info(f"{status} {filename} ({progress:.0f}%)")

    def log_batch_complete(self, successful: int, failed: int):
        """Log batch completion."""
        total = successful + failed
        self.logger.info(
            f"{self.EMOJI_MAP['success']} Completed: {successful}/{total} files"
        )
        if failed > 0:
            self.logger.warning(f"{self.EMOJI_MAP['warning']} Failed: {failed} files")

    def log_model_loading(self, model_name: str):
        """Log model loading."""
        self.logger.info(
            f"{self.EMOJI_MAP['processing']} Loading {model_name} model..."
        )

    def log_model_loaded(self, model_name: str):
        """Log successful model loading."""
        self.logger.info(
            f"{self.EMOJI_MAP['success']} {model_name} model loaded successfully"
        )

    def log_system_info(self, cpu_count: int, available_memory_gb: float):
        """Log system information."""
        self.logger.info(
            f"{self.EMOJI_MAP['info']} System: {cpu_count} CPU cores, {available_memory_gb:.1f} GB available memory"
        )
