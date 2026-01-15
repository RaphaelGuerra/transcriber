"""
Preflight checks for the transcriber application.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from logger import get_logger


def get_whisper_cache_dir() -> Path:
    base = os.getenv("XDG_CACHE_HOME")
    if not base:
        base = str(Path.home() / ".cache")
    return Path(base) / "whisper"


def _log_device_notice(logger):
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown GPU"
            logger.info(f"✅ GPU available: {gpu_count} device(s) - {gpu_name}")
        else:
            logger.warning("⚠️  CUDA not available; running on CPU")
    except Exception as exc:
        logger.warning(f"Hardware check failed: {exc}")


def run_preflight(config, logger: Optional[object] = None) -> bool:
    logger = logger or get_logger("preflight")
    ok = True

    issues = config.validate()
    if issues:
        logger.warning("Configuration issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.error("❌ FFmpeg not found in PATH. Install ffmpeg to process media files.")
        ok = False
    else:
        logger.info(f"🎬 FFmpeg detected: {ffmpeg_path}")

    cache_dir = get_whisper_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(cache_dir, os.W_OK):
            logger.error(f"❌ Whisper cache dir not writable: {cache_dir}")
            ok = False
        else:
            logger.info(f"🗂️  Whisper cache: {cache_dir}")
    except Exception as exc:
        logger.error(f"❌ Failed to prepare whisper cache dir {cache_dir}: {exc}")
        ok = False

    _log_device_notice(logger)

    if not ok:
        logger.error("Preflight failed. Fix the issues above and retry.")
    return ok
