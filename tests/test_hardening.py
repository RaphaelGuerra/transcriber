import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cli import CLI, CLIArgs
from config import TranscriberConfig
from file_handler import FileHandler
from main import process_files_with_mode


class _FakeTranscriber:
    def __init__(self):
        self.called = None

    def process_files_sequential(self, files, model_name):
        self.called = ("sequential", len(files), model_name, None)
        return 1, []

    def process_files_parallel(self, files, model_name, max_workers):
        self.called = ("parallel", len(files), model_name, max_workers)
        return len(files), []


class _FakeDaemon:
    def __init__(self, _config):
        pass

    def status(self):
        return {
            "running": True,
            "pid": 1234,
            "cpu_percent": 12.5,
            "status": "running",
            "started": 1,
        }


def _make_args(parallel=None, workers=None, daemon_action=None):
    return CLIArgs(
        model="base",
        parallel=parallel,
        workers=workers,
        input_dir=None,
        output_dir=None,
        output_format=None,
        config_file=None,
        log_level="INFO",
        log_file=None,
        files=[],
        verbose=False,
        batch_mode=False,
        resume=False,
        daemon_action=daemon_action,
        foreground=False,
        list_files=False,
    )


class TestHardening(unittest.TestCase):
    def test_relative_config_paths_become_absolute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                config = TranscriberConfig(
                    input_dir=Path("input_media"),
                    output_dir=Path("output_transcriptions"),
                    temp_dir=Path("temp"),
                )
            finally:
                os.chdir(previous_cwd)

        self.assertTrue(config.input_dir.is_absolute())
        self.assertTrue(config.output_dir.is_absolute())
        self.assertTrue(config.temp_dir.is_absolute())

    def test_process_mode_uses_parallel_when_enabled(self):
        transcriber = _FakeTranscriber()
        config = TranscriberConfig(parallel_processing=True, max_workers=4)
        args = _make_args(parallel=None, workers=None)
        files = [Path("a.m4a"), Path("b.m4a")]

        process_files_with_mode(transcriber, files, "base", args, config)

        self.assertEqual(transcriber.called, ("parallel", 2, "base", 4))

    def test_process_mode_respects_sequential_flag(self):
        transcriber = _FakeTranscriber()
        config = TranscriberConfig(parallel_processing=True, max_workers=4)
        args = _make_args(parallel=False)
        files = [Path("a.m4a"), Path("b.m4a")]

        process_files_with_mode(transcriber, files, "base", args, config)

        self.assertEqual(transcriber.called, ("sequential", 2, "base", None))

    def test_process_mode_uses_sequential_for_single_file(self):
        transcriber = _FakeTranscriber()
        config = TranscriberConfig(parallel_processing=True, max_workers=4)
        args = _make_args(parallel=True, workers=2)
        files = [Path("a.m4a")]

        process_files_with_mode(transcriber, files, "base", args, config)

        self.assertEqual(transcriber.called, ("sequential", 1, "base", None))

    def test_probe_validation_rejects_unknown_codec(self):
        handler = FileHandler(Path("input_media"), Path("output_transcriptions"), Path("temp"))
        is_valid, error = handler.validate_probe_result(30.0, "unknown")
        self.assertFalse(is_valid)
        self.assertIn("codec", error.lower())

    def test_get_file_info_skips_hash_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            handler = FileHandler(tmp / "in", tmp / "out", tmp / "tmp")
            file_path = tmp / "in" / "memo.m4a"
            file_path.write_bytes(b"demo")

            info_without_hash = handler.get_file_info(file_path)
            info_with_hash = handler.get_file_info(file_path, include_hash=True)

        self.assertNotIn("hash", info_without_hash)
        self.assertIn("hash", info_with_hash)

    def test_select_from_list_renders_items(self):
        cli = CLI()
        output = io.StringIO()
        with patch("builtins.input", return_value="1"), redirect_stdout(output):
            selected = cli.select_from_list(["alpha", "beta"], "Pick one", allow_multiple=False)

        rendered = output.getvalue()
        self.assertEqual(selected, ["alpha"])
        self.assertIn(" 1. alpha", rendered)
        self.assertIn(" 2. beta", rendered)

    def test_daemon_status_output_has_no_stray_format_text(self):
        cli = CLI()
        args = _make_args(daemon_action="status")
        output = io.StringIO()
        fake_daemon_module = types.ModuleType("daemon")
        fake_daemon_module.TranscriberDaemon = _FakeDaemon

        with patch.dict(sys.modules, {"daemon": fake_daemon_module}), redirect_stdout(output):
            cli.handle_daemon_command(args, TranscriberConfig())

        rendered = output.getvalue()
        self.assertNotIn(".1f", rendered)
        self.assertIn("CPU Usage: 12.5%", rendered)

    def test_setup_py_modules_include_runtime_dependencies(self):
        setup_path = Path(__file__).resolve().parents[1] / "setup.py"
        content = setup_path.read_text(encoding="utf-8")
        self.assertIn('"media_pipeline"', content)
        self.assertIn('"preflight"', content)
        self.assertIn('"subtitle_formats"', content)


if __name__ == "__main__":
    unittest.main()
