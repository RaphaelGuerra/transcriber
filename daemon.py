#!/usr/bin/env python3
"""
Background daemon service for the transcriber application.

This module provides functionality to run the transcriber as a background service
that can handle system sleep/wake cycles and continue processing automatically.
"""

import atexit
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import psutil

from config import TranscriberConfig, get_config
from logger import get_logger, setup_logging
from transcriber_core import TranscriberCore


class TranscriberDaemon:
    """Daemon service for background transcription processing."""

    def __init__(self, config: Optional[TranscriberConfig] = None, pid_file: Optional[str] = None):
        self.config = config or get_config()
        self.logger = get_logger("daemon")

        # Set up PID file
        self.pid_file = Path(pid_file or self.config.temp_dir / "transcriber_daemon.pid")

        # Initialize transcriber
        self.transcriber = TranscriberCore(self.config)

        # Daemon state
        self.running = False
        self.processing = False

        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGHUP, self._signal_handler)

        # Register cleanup
        atexit.register(self.cleanup)

    def _signal_handler(self, signum, frame):
        """Handle system signals."""
        sig_names = {
            signal.SIGTERM: "SIGTERM",
            signal.SIGINT: "SIGINT",
            signal.SIGHUP: "SIGHUP"
        }
        self.logger.info(f"Received signal: {sig_names.get(signum, signum)}")
        self.running = False

    def _write_pid_file(self):
        """Write the current process ID to the PID file."""
        try:
            self.pid_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
            self.logger.info(f"PID file written: {self.pid_file}")
        except Exception as e:
            self.logger.error(f"Failed to write PID file: {e}")
            raise

    def _remove_pid_file(self):
        """Remove the PID file."""
        try:
            if self.pid_file.exists():
                self.pid_file.unlink()
                self.logger.info(f"PID file removed: {self.pid_file}")
        except Exception as e:
            self.logger.error(f"Failed to remove PID file: {e}")

    def _check_existing_daemon(self) -> bool:
        """Check if another daemon instance is already running."""
        if not self.pid_file.exists():
            return False

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Check if process is actually running
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                if process.name() in ['python', 'python3'] and 'transcriber' in ' '.join(process.cmdline()):
                    self.logger.warning(f"Daemon already running with PID {pid}")
                    return True
            else:
                self.logger.warning(f"Stale PID file found, removing: {self.pid_file}")
                self._remove_pid_file()

        except (FileNotFoundError, ProcessLookupError, psutil.NoSuchProcess):
            self.logger.warning(f"Removing stale PID file: {self.pid_file}")
            self._remove_pid_file()
        except Exception as e:
            self.logger.error(f"Error checking existing daemon: {e}")

        return False

    def _daemonize(self):
        """Daemonize the current process."""
        try:
            # First fork (detaches from parent)
            pid = os.fork()
            if pid > 0:
                sys.exit(0)  # Parent exits

            # Create new session
            os.setsid()

            # Second fork (prevents acquiring controlling terminal)
            pid = os.fork()
            if pid > 0:
                sys.exit(0)  # First child exits

            # Change working directory
            os.chdir('/')

            # Close all open file descriptors
            for fd in range(0, 1024):
                try:
                    os.close(fd)
                except OSError:
                    pass

            # Redirect stdin, stdout, stderr to /dev/null or log files
            sys.stdin = open('/dev/null', 'r')
            sys.stdout = open('/dev/null', 'w')
            sys.stderr = open('/dev/null', 'w')

        except Exception as e:
            self.logger.error(f"Failed to daemonize: {e}")
            raise

    def _setup_system_monitoring(self):
        """Set up monitoring for system sleep/wake events."""
        try:
            import platform
            system = platform.system().lower()

            if system == 'darwin':  # macOS
                self.logger.info("macOS sleep monitoring disabled (causing false positives)")
                # self._setup_macos_monitoring()  # Temporarily disabled
            elif system == 'linux':
                self._setup_linux_monitoring()
            elif system == 'windows':
                self._setup_windows_monitoring()
            else:
                self.logger.warning(f"System monitoring not supported for {system}")

        except ImportError as e:
            self.logger.warning(f"System monitoring setup failed: {e}")

    def _setup_macos_monitoring(self):
        """Set up macOS-specific sleep/wake monitoring."""
        try:
            import subprocess
            import threading

            def monitor_sleep_events():
                cmd = ['pmset', '-g', 'log']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                while self.running:
                    if process.poll() is not None:
                        break

                    line = process.stdout.readline()
                    if 'Entering Sleep' in line:
                        self.logger.info("System entering sleep mode")
                        self.processing = False
                    elif 'Wake from' in line:
                        self.logger.info("System waking from sleep")
                        # Resume processing after a short delay
                        time.sleep(5)
                        self.processing = True

                process.terminate()

            thread = threading.Thread(target=monitor_sleep_events, daemon=True)
            thread.start()
            self.logger.info("macOS sleep/wake monitoring started")

        except Exception as e:
            self.logger.error(f"Failed to setup macOS monitoring: {e}")

    def _setup_linux_monitoring(self):
        """Set up Linux-specific sleep/wake monitoring."""
        try:
            import subprocess
            import threading

            def monitor_sleep_events():
                # Use systemd logind or acpid to monitor sleep events
                try:
                    cmd = ['journalctl', '-f', '-u', 'systemd-logind']
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    while self.running:
                        if process.poll() is not None:
                            break

                        line = process.stdout.readline()
                        if 'Entering sleep' in line.lower():
                            self.logger.info("System entering sleep mode")
                            self.processing = False
                        elif 'Resuming from' in line.lower():
                            self.logger.info("System waking from sleep")
                            time.sleep(5)
                            self.processing = True

                    process.terminate()

                except FileNotFoundError:
                    self.logger.warning("systemd not available for sleep monitoring")

            thread = threading.Thread(target=monitor_sleep_events, daemon=True)
            thread.start()
            self.logger.info("Linux sleep/wake monitoring started")

        except Exception as e:
            self.logger.error(f"Failed to setup Linux monitoring: {e}")

    def _setup_windows_monitoring(self):
        """Set up Windows-specific sleep/wake monitoring."""
        try:
            import threading
            import win32api
            import win32con

            def monitor_sleep_events():
                def window_proc(hwnd, msg, wparam, lparam):
                    if msg == win32con.WM_POWERBROADCAST:
                        if wparam == win32con.PBT_APMSUSPEND:
                            self.logger.info("System entering sleep mode")
                            self.processing = False
                        elif wparam == win32con.PBT_APMRESUMESUSPEND:
                            self.logger.info("System waking from sleep")
                            time.sleep(5)
                            self.processing = True
                    return 0

                import win32gui
                wc = win32gui.WNDCLASS()
                wc.lpfnWndProc = window_proc
                wc.lpszClassName = "PowerMonitor"
                class_atom = win32gui.RegisterClass(wc)

                hwnd = win32gui.CreateWindow(
                    class_atom, "PowerMonitor", 0, 0, 0, 0, 0, 0, 0, None, None
                )

                while self.running:
                    win32gui.PumpWaitingMessages()
                    time.sleep(0.1)

            thread = threading.Thread(target=monitor_sleep_events, daemon=True)
            thread.start()
            self.logger.info("Windows sleep/wake monitoring started")

        except ImportError:
            self.logger.warning("Windows monitoring requires pywin32")

    def _scan_and_process_files(self):
        """Main processing loop for the daemon."""
        while self.running:
            try:
                if not self.processing:
                    time.sleep(10)  # Wait while system is sleeping
                    continue

                # Get files to process
                files = self.transcriber.get_media_files()
                if not files:
                    self.logger.debug("No files to process, sleeping...")
                    time.sleep(30)  # Wait before next scan
                    continue

                # Filter out already processed files
                unprocessed_files = self._get_unprocessed_files(files)
                if not unprocessed_files:
                    self.logger.debug("All files already processed, sleeping...")
                    time.sleep(60)  # Wait longer if nothing to do
                    continue

                self.logger.info(f"Found {len(unprocessed_files)} files to process")

                # Process files
                for file_path in unprocessed_files:
                    if not self.running or not self.processing:
                        break

                    self.logger.info(f"Processing: {file_path.name}")

                    # Use sequential processing for daemon mode
                    result = self.transcriber.transcribe_file(
                        file_path, self.config.default_model
                    )

                    if result.success:
                        self.logger.info(f"Successfully processed: {file_path.name}")
                        self._mark_file_processed(file_path)
                    else:
                        self.logger.error(f"Failed to process: {file_path.name} - {result.error_message}")

                    # Small delay between files
                    time.sleep(2)

                # Wait before next scan
                time.sleep(30)

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(60)  # Wait longer on errors

    def _get_unprocessed_files(self, files):
        """Get list of files that haven't been processed yet."""
        processed_marker = self.config.temp_dir / "processed_files.txt"

        try:
            if processed_marker.exists():
                with open(processed_marker, 'r') as f:
                    processed = set(line.strip() for line in f)
            else:
                processed = set()
        except Exception as e:
            self.logger.error(f"Error reading processed files marker: {e}")
            processed = set()

        # Return files that haven't been processed
        return [f for f in files if str(f) not in processed]

    def _mark_file_processed(self, file_path):
        """Mark a file as processed."""
        processed_marker = self.config.temp_dir / "processed_files.txt"

        try:
            processed_marker.parent.mkdir(parents=True, exist_ok=True)
            with open(processed_marker, 'a') as f:
                f.write(f"{file_path}\n")
        except Exception as e:
            self.logger.error(f"Error marking file as processed: {e}")

    def start(self, daemonize: bool = True):
        """Start the daemon service."""
        self.logger.info("Starting transcriber daemon...")

        # Check for existing daemon
        if self._check_existing_daemon():
            self.logger.error("Another daemon instance is already running")
            return False

        if daemonize:
            self.logger.info("Daemonizing process...")
            self._daemonize()

        # Write PID file
        self._write_pid_file()

        # Set up system monitoring
        self._setup_system_monitoring()

        # Start processing
        self.running = True
        self.processing = True

        self.logger.info("Transcriber daemon started successfully")
        self.logger.info(f"PID: {os.getpid()}")
        self.logger.info(f"Watching directory: {self.config.input_dir}")

        try:
            self._scan_and_process_files()
        except Exception as e:
            self.logger.error(f"Daemon error: {e}")
        finally:
            self.cleanup()

        return True

    def stop(self):
        """Stop the daemon service."""
        if not self.pid_file.exists():
            self.logger.warning("No PID file found - daemon may not be running")
            return False

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            # Send SIGTERM to the daemon process
            os.kill(pid, signal.SIGTERM)
            self.logger.info(f"Sent SIGTERM to daemon process {pid}")

            # Wait for process to terminate
            for _ in range(30):  # Wait up to 30 seconds
                if not psutil.pid_exists(pid):
                    break
                time.sleep(1)

            if psutil.pid_exists(pid):
                self.logger.warning(f"Daemon process {pid} did not terminate gracefully")
                return False

            return True

        except FileNotFoundError:
            self.logger.warning("PID file not found")
            return False
        except ProcessLookupError:
            self.logger.warning("Daemon process not found")
            return False
        except Exception as e:
            self.logger.error(f"Error stopping daemon: {e}")
            return False

    def status(self):
        """Get the status of the daemon service."""
        if not self.pid_file.exists():
            return {"running": False, "message": "No PID file found"}

        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())

            if not psutil.pid_exists(pid):
                return {"running": False, "message": "Process not found"}

            process = psutil.Process(pid)
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()

            return {
                "running": True,
                "pid": pid,
                "cpu_percent": cpu_percent,
                "memory_mb": memory_info.rss / 1024 / 1024,
                "status": process.status(),
                "started": process.create_time()
            }

        except Exception as e:
            return {"running": False, "message": f"Error: {e}"}

    def cleanup(self):
        """Clean up daemon resources."""
        self.logger.info("Cleaning up daemon resources...")
        self.running = False
        self.processing = False
        self._remove_pid_file()


def main():
    """Main entry point for daemon operations."""
    import argparse

    parser = argparse.ArgumentParser(description="Transcriber Daemon Control")
    parser.add_argument(
        "action", choices=["start", "stop", "restart", "status"],
        help="Action to perform"
    )
    parser.add_argument(
        "--foreground", "-f", action="store_true",
        help="Run in foreground (don't daemonize)"
    )
    parser.add_argument(
        "--config", type=Path,
        help="Configuration file path"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Set logging level"
    )
    parser.add_argument(
        "--log-file", type=Path,
        help="Log file path"
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        console=not args.foreground,
        color=True
    )

    # Load configuration
    config = get_config(args.config)

    # Create daemon instance
    daemon = TranscriberDaemon(config)

    if args.action == "start":
        if not daemon.start(daemonize=not args.foreground):
            sys.exit(1)

    elif args.action == "stop":
        if not daemon.stop():
            sys.exit(1)

    elif args.action == "restart":
        daemon.stop()
        time.sleep(2)  # Wait for cleanup
        if not daemon.start(daemonize=not args.foreground):
            sys.exit(1)

    elif args.action == "status":
        status = daemon.status()
        if status["running"]:
            print("✅ Daemon is running")
            print(f"   PID: {status['pid']}")
            print(f"   CPU: {status['cpu_percent']:.1f}%")
            print(".1f")
            print(f"   Status: {status['status']}")
            print(f"   Started: {time.ctime(status['started'])}")
        else:
            print("❌ Daemon is not running")
            print(f"   Message: {status['message']}")


if __name__ == "__main__":
    main()
