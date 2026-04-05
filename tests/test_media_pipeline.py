import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from config import TranscriberConfig
from media_pipeline import (
    AudioChunk,
    MediaProbeInfo,
    PreparedAudio,
    build_chunk_plan,
    merge_chunk_results,
    parse_probe_output,
    prepare_audio_for_transcription,
    probe_media,
)
from transcriber_core import TranscriberCore


class FakeModel:
    def transcribe(self, path, verbose=False):
        name = Path(path).name
        if "short" in name or "normalized" in name:
            return {
                "text": "hello from short memo",
                "segments": [{"start": 0.0, "end": 2.0, "text": "hello from short memo"}],
            }
        if "chunk_001" in name:
            return {
                "text": "hello there general",
                "segments": [
                    {"start": 0.0, "end": 4.0, "text": "hello there"},
                    {"start": 1196.0, "end": 1200.0, "text": "general"},
                ],
            }
        if "chunk_002" in name:
            return {
                "text": "general kenobi",
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "general"},
                    {"start": 2.0, "end": 6.0, "text": "kenobi"},
                ],
            }
        return {"text": "fallback", "segments": []}


class TestMediaPipeline(unittest.TestCase):
    def test_parse_probe_output(self):
        payload = {
            "streams": [
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "duration": "123.45",
                    "sample_rate": "44100",
                    "channels": 2,
                }
            ],
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2", "duration": "123.45"},
        }

        info = parse_probe_output(payload, Path("memo.m4a"))

        self.assertEqual(info.codec_name, "aac")
        self.assertEqual(info.container, "mov,mp4,m4a,3gp,3g2,mj2")
        self.assertEqual(info.sample_rate_hz, 44100)
        self.assertEqual(info.channels, 2)
        self.assertAlmostEqual(info.duration_seconds, 123.45)

    def test_build_chunk_plan_uses_overlap(self):
        plan = build_chunk_plan(3900, chunk_duration_minutes=20, overlap_seconds=2)
        self.assertEqual(
            plan,
            [(0.0, 1200.0), (1198.0, 2398.0), (2396.0, 3596.0), (3594.0, 3900.0)],
        )

    def test_merge_chunk_results_deduplicates_overlap(self):
        chunk_one = AudioChunk(Path("chunk_001.wav"), 0.0, 1200.0, 1)
        chunk_two = AudioChunk(Path("chunk_002.wav"), 1198.0, 2398.0, 2)

        merged = merge_chunk_results(
            [
                (
                    chunk_one,
                    {
                        "text": "hello there general",
                        "segments": [
                            {"start": 0.0, "end": 4.0, "text": "hello there"},
                            {"start": 1196.0, "end": 1200.0, "text": "general"},
                        ],
                    },
                ),
                (
                    chunk_two,
                    {
                        "text": "general kenobi",
                        "segments": [
                            {"start": 0.0, "end": 2.0, "text": "general"},
                            {"start": 2.0, "end": 6.0, "text": "kenobi"},
                        ],
                    },
                ),
            ]
        )

        self.assertEqual(merged["text"], "hello there general kenobi")
        self.assertEqual(len(merged["segments"]), 3)
        self.assertEqual(merged["segments"][2]["start"], 1200.0)
        self.assertEqual(merged["segments"][2]["end"], 1204.0)


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "ffmpeg/ffprobe required for integration test",
)
class TestMediaPipelineIntegration(unittest.TestCase):
    def test_prepare_audio_for_transcription_normalizes_voice_memo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            wav_path = tmp / "source.wav"
            m4a_path = tmp / "memo.m4a"

            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 16000)

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(wav_path),
                    str(m4a_path),
                ],
                check=True,
            )

            probe = probe_media(m4a_path)
            prepared = prepare_audio_for_transcription(m4a_path, tmp, probe)
            try:
                self.assertTrue(prepared.normalized_path.exists())
                self.assertEqual(prepared.normalized_path.suffix, ".wav")
                self.assertEqual(len(prepared.chunks), 1)
            finally:
                prepared.cleanup()


class TestTranscriberSmoke(unittest.TestCase):
    def test_process_files_sequential_handles_short_and_long_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "input"
            output_dir = root / "output"
            temp_dir = root / "temp"
            input_dir.mkdir()
            output_dir.mkdir()
            temp_dir.mkdir()

            short_file = input_dir / "short.m4a"
            long_file = input_dir / "long.m4a"
            short_file.write_bytes(b"short")
            long_file.write_bytes(b"long")

            config = TranscriberConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                temp_dir=temp_dir,
                include_metadata=False,
                output_format="txt",
            )
            transcriber = TranscriberCore(config)

            short_probe = MediaProbeInfo(
                path=short_file,
                container="mov,mp4,m4a,3gp,3g2,mj2",
                codec_name="aac",
                duration_seconds=60.0,
                sample_rate_hz=44100,
                channels=1,
            )
            long_probe = MediaProbeInfo(
                path=long_file,
                container="mov,mp4,m4a,3gp,3g2,mj2",
                codec_name="aac",
                duration_seconds=2400.0,
                sample_rate_hz=44100,
                channels=1,
            )

            def fake_probe(path):
                return short_probe if path == short_file else long_probe

            def fake_prepare(file_path, temp_dir, probe_info, **_kwargs):
                workspace = temp_dir / f"{file_path.stem}_workspace"
                workspace.mkdir(exist_ok=True)
                normalized = workspace / "normalized.wav"
                normalized.write_bytes(b"wav")
                if file_path == short_file:
                    chunks = [AudioChunk(normalized, 0.0, 60.0, 1)]
                else:
                    chunk_one = workspace / "chunk_001.wav"
                    chunk_two = workspace / "chunk_002.wav"
                    chunk_one.write_bytes(b"chunk1")
                    chunk_two.write_bytes(b"chunk2")
                    chunks = [
                        AudioChunk(chunk_one, 0.0, 1200.0, 1),
                        AudioChunk(chunk_two, 1198.0, 2398.0, 2),
                    ]
                return PreparedAudio(
                    source_path=file_path,
                    probe_info=probe_info,
                    normalized_path=normalized,
                    chunks=chunks,
                    workspace_dir=workspace,
                )

            with patch("transcriber_core.probe_media", side_effect=fake_probe), patch(
                "transcriber_core.prepare_audio_for_transcription",
                side_effect=fake_prepare,
            ), patch.object(TranscriberCore, "load_model", return_value=FakeModel()):
                successful, failed = transcriber.process_files_sequential(
                    [short_file, long_file], "base"
                )

            self.assertEqual(successful, 2)
            self.assertEqual(failed, [])
            outputs = sorted(output_dir.glob("*.txt"))
            self.assertEqual(len(outputs), 2)
            contents = [output.read_text(encoding="utf-8") for output in outputs]
            self.assertTrue(any("hello from short memo" in content for content in contents))
            self.assertTrue(any("hello there general kenobi" in content for content in contents))


if __name__ == "__main__":
    unittest.main()
