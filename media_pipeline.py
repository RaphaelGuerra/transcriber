"""Media probing, normalization, chunking, and merge helpers."""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class MediaProcessingError(Exception):
    """Raised when media probing or preprocessing fails."""


@dataclass
class MediaProbeInfo:
    """Normalized audio probe information."""

    path: Path
    container: str
    codec_name: str
    duration_seconds: float
    sample_rate_hz: Optional[int]
    channels: Optional[int]


@dataclass
class AudioChunk:
    """Prepared chunk of audio ready for Whisper."""

    path: Path
    start_seconds: float
    end_seconds: float
    index: int


@dataclass
class PreparedAudio:
    """Prepared audio artifact(s) used for transcription."""

    source_path: Path
    probe_info: MediaProbeInfo
    normalized_path: Path
    chunks: List[AudioChunk]
    workspace_dir: Optional[Path] = None

    def cleanup(self) -> None:
        """Remove generated temporary files."""
        if self.workspace_dir and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)


def parse_probe_output(payload: Dict[str, Any], source_path: Path) -> MediaProbeInfo:
    """Parse `ffprobe` JSON output into a normalized structure."""
    streams = payload.get("streams") or []
    format_info = payload.get("format") or {}
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]

    if not audio_streams:
        raise MediaProcessingError(f"No audio stream found in {source_path.name}")

    audio_stream = audio_streams[0]
    duration_raw = audio_stream.get("duration") or format_info.get("duration")
    if duration_raw in (None, "", "N/A"):
        raise MediaProcessingError(f"Unable to determine duration for {source_path.name}")

    try:
        duration_seconds = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise MediaProcessingError(
            f"Invalid duration '{duration_raw}' for {source_path.name}"
        ) from exc

    if duration_seconds <= 0:
        raise MediaProcessingError(f"Non-positive duration reported for {source_path.name}")

    sample_rate_raw = audio_stream.get("sample_rate")
    channels_raw = audio_stream.get("channels")

    return MediaProbeInfo(
        path=source_path,
        container=format_info.get("format_name") or "unknown",
        codec_name=audio_stream.get("codec_name") or "unknown",
        duration_seconds=duration_seconds,
        sample_rate_hz=int(sample_rate_raw) if sample_rate_raw not in (None, "", "N/A") else None,
        channels=int(channels_raw) if channels_raw not in (None, "", "N/A") else None,
    )


def probe_media(file_path: Path, ffprobe_path: str = "ffprobe") -> MediaProbeInfo:
    """Probe media metadata with `ffprobe`."""
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MediaProcessingError(
            "ffprobe is required to inspect audio files. Install FFmpeg (which includes ffprobe)."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() or "unknown ffprobe error"
        raise MediaProcessingError(
            f"ffprobe could not read {file_path.name}: {stderr}"
        ) from exc

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaProcessingError(
            f"ffprobe returned invalid JSON for {file_path.name}"
        ) from exc

    return parse_probe_output(payload, file_path)


def build_chunk_plan(
    duration_seconds: float,
    chunk_duration_minutes: int,
    overlap_seconds: int,
) -> List[Tuple[float, float]]:
    """Build a duration-based chunk plan with overlap."""
    if duration_seconds <= 0:
        return []

    chunk_duration_seconds = max(1, int(chunk_duration_minutes * 60))
    overlap_seconds = max(0, int(overlap_seconds))

    if duration_seconds <= chunk_duration_seconds:
        return [(0.0, float(duration_seconds))]

    plan: List[Tuple[float, float]] = []
    start_seconds = 0.0

    while start_seconds < duration_seconds:
        end_seconds = min(duration_seconds, start_seconds + chunk_duration_seconds)
        plan.append((round(start_seconds, 3), round(end_seconds, 3)))

        if end_seconds >= duration_seconds:
            break

        next_start = max(0.0, end_seconds - overlap_seconds)
        if next_start <= start_seconds:
            next_start = end_seconds
        start_seconds = next_start

    return plan


def normalize_audio(
    source_path: Path,
    normalized_path: Path,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """Normalize audio to mono 16 kHz PCM WAV for reliable transcription."""
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(normalized_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise MediaProcessingError(
            "ffmpeg is required to normalize audio. Install FFmpeg and retry."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() or "unknown ffmpeg error"
        raise MediaProcessingError(
            f"ffmpeg could not normalize {source_path.name}: {stderr}"
        ) from exc

    return normalized_path


def split_audio_chunks(
    source_path: Path,
    chunk_plan: Sequence[Tuple[float, float]],
    workspace_dir: Path,
    ffmpeg_path: str = "ffmpeg",
) -> List[AudioChunk]:
    """Split a prepared audio file into timestamped WAV chunks."""
    chunks: List[AudioChunk] = []

    for index, (start_seconds, end_seconds) in enumerate(chunk_plan, start=1):
        duration_seconds = max(0.0, end_seconds - start_seconds)
        if duration_seconds <= 0:
            continue

        chunk_path = workspace_dir / f"chunk_{index:03d}.wav"
        command = [
            ffmpeg_path,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(chunk_path),
        ]

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise MediaProcessingError(
                "ffmpeg is required to split long audio. Install FFmpeg and retry."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip() or "unknown ffmpeg error"
            raise MediaProcessingError(
                f"ffmpeg could not split chunk {index} from {source_path.name}: {stderr}"
            ) from exc

        chunks.append(
            AudioChunk(
                path=chunk_path,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                index=index,
            )
        )

    return chunks


def prepare_audio_for_transcription(
    file_path: Path,
    temp_dir: Path,
    probe_info: MediaProbeInfo,
    preprocess_audio: bool = True,
    long_audio_threshold_minutes: int = 30,
    chunk_duration_minutes: int = 20,
    chunk_overlap_seconds: int = 2,
) -> PreparedAudio:
    """Normalize and optionally chunk audio before transcription."""
    workspace_dir = Path(
        tempfile.mkdtemp(prefix=f"{file_path.stem}_", dir=str(temp_dir))
    )

    needs_normalization = preprocess_audio or file_path.suffix.lower() != ".wav"
    if not needs_normalization:
        normalized_path = file_path
    else:
        normalized_path = workspace_dir / "normalized.wav"
        normalize_audio(file_path, normalized_path)

    threshold_seconds = max(1, int(long_audio_threshold_minutes * 60))
    if probe_info.duration_seconds > threshold_seconds:
        plan = build_chunk_plan(
            probe_info.duration_seconds,
            chunk_duration_minutes=chunk_duration_minutes,
            overlap_seconds=chunk_overlap_seconds,
        )
        chunks = split_audio_chunks(normalized_path, plan, workspace_dir)
    else:
        chunks = [
            AudioChunk(
                path=normalized_path,
                start_seconds=0.0,
                end_seconds=probe_info.duration_seconds,
                index=1,
            )
        ]

    return PreparedAudio(
        source_path=file_path,
        probe_info=probe_info,
        normalized_path=normalized_path,
        chunks=chunks,
        workspace_dir=workspace_dir if workspace_dir.exists() else None,
    )


def merge_chunk_results(
    chunk_results: Iterable[Tuple[AudioChunk, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Merge chunk-local Whisper results into one timeline."""
    merged_segments: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    last_end_seconds = 0.0

    for chunk, result in chunk_results:
        segments = result.get("segments") or []
        if not segments:
            chunk_text = (result.get("text") or "").strip()
            if chunk_text:
                text_parts.append(chunk_text)
            continue

        for segment in segments:
            text = (segment.get("text") or "").strip()
            if not text:
                continue

            start_seconds = max(0.0, float(segment.get("start", 0.0))) + chunk.start_seconds
            end_seconds = max(start_seconds, float(segment.get("end", 0.0)) + chunk.start_seconds)

            if end_seconds <= last_end_seconds + 0.05:
                continue

            if start_seconds < last_end_seconds:
                start_seconds = last_end_seconds

            merged_segment = dict(segment)
            merged_segment["start"] = round(start_seconds, 3)
            merged_segment["end"] = round(end_seconds, 3)
            merged_segment["text"] = text
            merged_segments.append(merged_segment)
            text_parts.append(text)
            last_end_seconds = max(last_end_seconds, end_seconds)

    merged_text = " ".join(part for part in text_parts if part).strip()
    return {"text": merged_text, "segments": merged_segments}
