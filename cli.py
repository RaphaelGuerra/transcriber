"""
Command-line interface for the transcriber application
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from config import TranscriberConfig, get_config
from logger import get_logger, setup_logging


@dataclass
class CLIArgs:
    """Container for parsed command-line arguments."""

    model: str
    parallel: Optional[bool]
    workers: Optional[int]
    input_dir: Optional[Path]
    output_dir: Optional[Path]
    config_file: Optional[Path]
    log_level: str
    log_file: Optional[Path]
    files: List[str]
    verbose: bool
    batch_mode: bool
    resume: bool
    daemon_action: Optional[str]
    foreground: bool
    list_files: bool


class CLI:
    """Command-line interface handler."""

    def __init__(self):
        self.logger = get_logger("cli")

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="transcriber",
            description="Fast Video/Audio Transcriber using OpenAI Whisper",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Interactive mode (default)
  transcriber

  # Transcribe specific files with base model
  transcriber --files video1.mp4 audio1.mp3 --model base

  # Batch process with parallel workers
  transcriber --model medium --parallel --workers 4

  # Use custom directories
  transcriber --input-dir /path/to/files --output-dir /path/to/transcriptions

  # Enable logging
  transcriber --log-level DEBUG --log-file transcriber.log
            """,
        )

        # Model selection
        parser.add_argument(
            "--model",
            "-m",
            choices=["tiny", "base", "small", "medium", "large"],
            default="base",
            help="Whisper model to use (default: base)",
        )

        # Processing options
        processing_group = parser.add_argument_group("Processing Options")
        processing_group.add_argument(
            "--parallel", "-p", action="store_true", help="Use parallel processing"
        )
        processing_group.add_argument(
            "--sequential",
            action="store_true",
            help="Use sequential processing (default for single files)",
        )
        processing_group.add_argument(
            "--workers",
            "-w",
            type=int,
            metavar="N",
            help="Number of parallel workers (auto-detected if not specified)",
        )

        # Directory options
        dir_group = parser.add_argument_group("Directory Options")
        dir_group.add_argument(
            "--input-dir",
            "-i",
            type=Path,
            metavar="PATH",
            help="Input directory for media files (default: input_media)",
        )
        dir_group.add_argument(
            "--output-dir",
            "-o",
            type=Path,
            metavar="PATH",
            help="Output directory for transcriptions (default: output_transcriptions)",
        )

        # Configuration
        config_group = parser.add_argument_group("Configuration")
        config_group.add_argument(
            "--config", type=Path, metavar="FILE", help="Configuration file path"
        )

        # Logging options
        logging_group = parser.add_argument_group("Logging Options")
        logging_group.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Set the logging level (default: INFO)",
        )
        logging_group.add_argument(
            "--log-file", type=Path, metavar="FILE", help="Log file path"
        )
        logging_group.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Verbose output (equivalent to --log-level DEBUG)",
        )

        # File selection
        file_group = parser.add_argument_group("File Selection")
        file_group.add_argument(
            "--files",
            nargs="+",
            metavar="FILE",
            help="Specific files to process (overrides directory scanning)",
        )

        # Operational modes
        mode_group = parser.add_argument_group("Operational Modes")
        mode_group.add_argument(
            "--batch", action="store_true", help="Run in batch mode (non-interactive)"
        )
        mode_group.add_argument(
            "--resume", action="store_true", help="Resume interrupted transcription"
        )

        # Daemon commands
        daemon_group = parser.add_argument_group("Daemon Control")
        daemon_group.add_argument(
            "--daemon",
            choices=["start", "stop", "restart", "status"],
            help="Control the background daemon service"
        )
        daemon_group.add_argument(
            "--foreground", "-f", action="store_true",
            help="Run daemon in foreground (for debugging)"
        )

        # Information commands
        info_group = parser.add_argument_group("Information")
        info_group.add_argument(
            "--list-models", action="store_true", help="List available models and exit"
        )
        info_group.add_argument(
            "--list-files",
            action="store_true",
            help="List available media files and exit",
        )
        info_group.add_argument("--version", action="version", version="%(prog)s 1.0.0")

        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> CLIArgs:
        """Parse command-line arguments."""
        parser = self.create_parser()
        parsed = parser.parse_args(args)

        # Handle special information commands
        if parsed.list_models:
            self._list_models()
            sys.exit(0)

        if parsed.list_files:
            self._list_files()
            sys.exit(0)

        # Adjust log level if verbose
        if parsed.verbose:
            parsed.log_level = "DEBUG"

        # Validate conflicting options
        if parsed.parallel and parsed.sequential:
            parser.error("--parallel and --sequential cannot be used together")

        if parsed.workers and not parsed.parallel:
            parser.error("--workers requires --parallel")

        # Set parallel mode based on flags
        parallel = None
        if parsed.parallel:
            parallel = True
        elif parsed.sequential:
            parallel = False

        return CLIArgs(
            model=parsed.model,
            parallel=parallel,
            workers=parsed.workers,
            input_dir=parsed.input_dir,
            output_dir=parsed.output_dir,
            config_file=parsed.config,
            log_level=parsed.log_level,
            log_file=parsed.log_file,
            files=parsed.files or [],
            verbose=parsed.verbose,
            batch_mode=parsed.batch,
            resume=parsed.resume,
            daemon_action=parsed.daemon,
            foreground=parsed.foreground,
            list_files=parsed.list_files,
        )

    def _list_models(self):
        """List available models."""
        models = {
            "tiny": "Fastest, lowest accuracy (~39 MB)",
            "base": "Good balance (default) (~74 MB)",
            "small": "Better accuracy (~244 MB)",
            "medium": "High accuracy (~769 MB)",
            "large": "Best accuracy (~1.5 GB)",
        }

        print("Available Whisper models:")
        print("=" * 50)
        for model, description in models.items():
            print(f"{model:15} {description}")

    def _list_files(self):
        """List available media files."""
        from transcriber_core import TranscriberCore
        from config import get_config

        transcriber = TranscriberCore(get_config())
        files = transcriber.get_media_files()

        if not files:
            print("❌ No media files found in input_media/")
            print("\nSupported formats: mp4, avi, mov, mkv, webm, flv, mp3, wav, m4a, aac, flac, ogg")
            return

        print(f"🎬 Found {len(files)} media file(s):")
        for i, file_path in enumerate(files, 1):
            file_info = transcriber.file_handler.get_file_info(file_path)
            print(f"{i:2d}. {file_path.name} ({file_info['size_mb']:.1f} MB)")

    def get_config_from_args(self, args: CLIArgs) -> TranscriberConfig:
        """Create configuration from command-line arguments."""
        config = get_config(args.config_file)

        # Override config with command-line arguments
        if args.model:
            config.default_model = args.model
        if args.input_dir:
            config.input_dir = args.input_dir
        if args.output_dir:
            config.output_dir = args.output_dir

        # Validate configuration
        issues = config.validate()
        if issues:
            self.logger.warning("Configuration issues found:")
            for issue in issues:
                self.logger.warning(f"  - {issue}")

        return config

    def setup_logging_from_args(self, args: CLIArgs):
        """Set up logging based on command-line arguments."""
        return setup_logging(
            log_level=args.log_level, log_file=args.log_file, console=True, color=True
        )

    def confirm_action(self, message: str, default: bool = True) -> bool:
        """Get user confirmation."""
        default_text = "Y/n" if default else "y/N"
        choice = input(f"{message} [{default_text}]: ").strip().lower()

        if not choice:
            return default

        return choice in ["y", "yes"]

    def select_from_list(
        self, items: List[str], prompt: str, allow_multiple: bool = False
    ) -> List[str]:
        """Let user select items from a list."""
        if not items:
            return []

        print(f"\n{prompt}")
        for i, item in enumerate(items, 1):
            print("2d")

        if allow_multiple:
            print(
                "\nEnter numbers separated by commas (e.g., 1,3,5) or 'all' for all items"
            )
            while True:
                choice = input("Selection: ").strip().lower()
                if choice == "all":
                    return items

                try:
                    indices = [int(x.strip()) - 1 for x in choice.split(",")]
                    selected = [items[i] for i in indices if 0 <= i < len(items)]
                    if selected:
                        return selected
                    else:
                        print("❌ No valid items selected")
                except (ValueError, IndexError):
                    print("❌ Invalid input. Use numbers like '1,3,5' or 'all'")
        else:
            while True:
                try:
                    choice = input(f"Select item (1-{len(items)}): ").strip()
                    index = int(choice) - 1
                    if 0 <= index < len(items):
                        return [items[index]]
                    else:
                        print(f"❌ Please enter a number between 1 and {len(items)}")
                except ValueError:
                    print("❌ Please enter a valid number")

    def handle_daemon_command(self, args: CLIArgs, config: TranscriberConfig):
        """Handle daemon control commands."""
        from daemon import TranscriberDaemon

        daemon = TranscriberDaemon(config)

        if args.daemon_action == "start":
            print("🚀 Starting transcriber daemon...")
            if daemon.start(daemonize=not args.foreground):
                if args.foreground:
                    print("✅ Daemon started in foreground mode")
                else:
                    print("✅ Daemon started in background mode")
            else:
                print("❌ Failed to start daemon")
                return False

        elif args.daemon_action == "stop":
            print("🛑 Stopping transcriber daemon...")
            if daemon.stop():
                print("✅ Daemon stopped successfully")
            else:
                print("❌ Failed to stop daemon")
                return False

        elif args.daemon_action == "restart":
            print("🔄 Restarting transcriber daemon...")
            if daemon.stop():
                print("   ✅ Daemon stopped")
            else:
                print("   ⚠️  Could not stop daemon (may not have been running)")

            import time
            time.sleep(2)

            if daemon.start(daemonize=not args.foreground):
                if args.foreground:
                    print("   ✅ Daemon restarted in foreground mode")
                else:
                    print("   ✅ Daemon restarted in background mode")
            else:
                print("   ❌ Failed to restart daemon")
                return False

        elif args.daemon_action == "status":
            status = daemon.status()
            if status["running"]:
                print("✅ Daemon is running")
                print(f"   PID: {status['pid']}")
                print(f"   CPU Usage: {status['cpu_percent']:.1f}%")
                print(".1f")
                print(f"   Status: {status['status']}")
                import time
                print(f"   Started: {time.ctime(status['started'])}")
            else:
                print("❌ Daemon is not running")
                print(f"   Status: {status['message']}")

        return True

    def display_welcome(self):
        """Display welcome message."""
        welcome_text = """
🎬 Fast Video/Audio Transcriber v1.0.0
========================================
A streamlined, high-performance tool for transcribing video and audio files using OpenAI's Whisper model.
        """
        print(welcome_text.strip())

    def display_menu(self, options: List[str]) -> int:
        """Display a menu and get user selection."""
        print("\n📋 Options:")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        while True:
            try:
                choice = input(f"\nSelect option (1-{len(options)}): ").strip()
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return index + 1
                else:
                    print(f"❌ Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("❌ Please enter a valid number")
