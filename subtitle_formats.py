"""Subtitle format helpers for Whisper segments."""

from typing import Iterable, Dict, List


def _format_timestamp(seconds: float, sep: str) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    total_ms %= 3_600_000
    minutes = total_ms // 60_000
    total_ms %= 60_000
    secs = total_ms // 1_000
    millis = total_ms % 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def format_timestamp_srt(seconds: float) -> str:
    return _format_timestamp(seconds, ",")


def format_timestamp_vtt(seconds: float) -> str:
    return _format_timestamp(seconds, ".")


def segments_to_srt(segments: Iterable[Dict]) -> str:
    lines: List[str] = []
    for idx, segment in enumerate(segments, 1):
        start = format_timestamp_srt(segment.get("start", 0))
        end = format_timestamp_srt(segment.get("end", 0))
        text = (segment.get("text") or "").strip()
        lines.append(str(idx))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def segments_to_vtt(segments: Iterable[Dict]) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for segment in segments:
        start = format_timestamp_vtt(segment.get("start", 0))
        end = format_timestamp_vtt(segment.get("end", 0))
        text = (segment.get("text") or "").strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
