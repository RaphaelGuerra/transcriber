#!/usr/bin/env python3
"""
Main entry point for the transcriber application.

This script handles both interactive and daemon modes of operation.
"""

import sys
from pathlib import Path

from cli import CLI, CLIArgs
from config import TranscriberConfig, get_config
from logger import setup_logging
from transcriber_core import TranscriberCore


def run_interactive_mode(transcriber: TranscriberCore):
    """Run the interactive mode for the transcriber."""
    while True:
        files = transcriber.show_files()

        if not files:
            print("\n💡 Place media files in 'input_media/' folder and run again")
            break

        print("\n📋 Options:")
        print("1. Transcribe files")
        print("2. View transcriptions")
        print("3. Exit")

        choice = input("\nChoice (1-3): ").strip()

        if choice == "1":
            # Model selection
            print("\n🎯 Model options:")
            models = ["tiny", "base", "small", "medium", "large"]
            for i, model in enumerate(models, 1):
                print(f"{i}. {model}")

            while True:
                try:
                    model_choice = input(f"Select model (1-5) [2]: ").strip()
                    if not model_choice:
                        model_choice = "2"

                    model_idx = int(model_choice) - 1
                    if 0 <= model_idx < len(models):
                        model_name = models[model_idx]
                        break
                    else:
                        print("❌ Invalid choice")
                except ValueError:
                    print("❌ Enter a number")

            # File selection
            print(f"\n🎬 Select files to transcribe:")
            print("Enter numbers separated by commas (e.g., 1,3,5) or 'all' for all files")

            while True:
                choice = input("Selection: ").strip().lower()

                if choice == "all":
                    selected_files = files
                    break

                try:
                    indices = [int(x.strip()) - 1 for x in choice.split(",")]
                    selected_files = [files[i] for i in indices if 0 <= i < len(files)]
                    if selected_files:
                        print(f"Selected: {len(selected_files)} file(s)")
                        break
                    else:
                        print("❌ No valid files selected")
                except (ValueError, IndexError):
                    print("❌ Invalid input. Use numbers like '1,3,5' or 'all'")

            # Processing mode selection
            parallel = False
            if len(selected_files) > 1:
                print(f"\n🚀 Processing mode selection:")
                print(f"   Files to process: {len(selected_files)}")
                print(f"   Available CPU cores: {mp.cpu_count()}")

                while True:
                    choice = input("\nUse parallel processing? (y/n) [y]: ").strip().lower()
                    if choice in ["", "y", "yes"]:
                        parallel = True
                        break
                    elif choice in ["n", "no"]:
                        parallel = False
                        break
                    else:
                        print("❌ Please enter 'y' or 'n'")

            # Process files
            if parallel:
                max_workers = min(mp.cpu_count(), len(selected_files))
                print(f"   Using {max_workers} parallel workers")
                transcriber.process_files_parallel(selected_files, model_name, max_workers)
            else:
                transcriber.process_files_sequential(selected_files, model_name)

            # Continue option
            if input("\n🔄 Process more files? (y/n): ").strip().lower() not in ["y", "yes"]:
                break

        elif choice == "2":
            transcriber.show_outputs()

        elif choice == "3":
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice")


def run_smart_mode(transcriber: TranscriberCore, config):
    """Smart auto-mode with interactive file and model selection."""
    # Check for existing files
    files = transcriber.get_media_files()

    if files:
        print(f"📁 Found {len(files)} file(s) ready for processing")
        print("🎯 Choose what to process:")
        print()

        # Show file selection menu
        print("📋 Available files:")
        for i, file_path in enumerate(files, 1):
            file_info = transcriber.file_handler.get_file_info(file_path)
            print(f"{i:2d}. {file_path.name} ({file_info['size_mb']:.1f} MB)")

        print("\nOptions:")
        print("• Enter numbers (e.g., '1,3,5') to select specific files")
        print("• Enter 'all' to process all files")
        print("• Enter 'menu' for full interactive mode")

        while True:
            try:
                choice = input("\nYour choice: ").strip().lower()

                if choice == 'menu':
                    run_interactive_mode(transcriber)
                    return

                if choice == 'all':
                    selected_files = files
                    break

                # Parse number selections
                indices = []
                for part in choice.split(','):
                    part = part.strip()
                    if part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(files):
                            indices.append(idx)
                        else:
                            print(f"❌ Invalid file number: {part}")
                            indices = []
                            break

                if indices:
                    selected_files = [files[i] for i in indices]
                    print(f"✅ Selected {len(selected_files)} file(s)")
                    break
                else:
                    print("❌ No valid selections. Try again.")

            except KeyboardInterrupt:
                print("\n\n👋 Cancelled")
                return
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

        # Show model selection
        print("\n🎯 Choose transcription model:")
        models = {
            'tiny': '⚡ Fastest (good quality)',
            'base': '⚖️  Balanced (default)',
            'small': '📈 Better accuracy',
            'medium': '🎯 High accuracy',
            'large': '🏆 Best accuracy (slowest)'
        }

        for i, (model, desc) in enumerate(models.items(), 1):
            default_marker = " (default)" if model == config.default_model else ""
            print(f"{i}. {model}{default_marker} - {desc}")

        while True:
            try:
                model_choice = input(f"\nSelect model (1-5) [2]: ").strip()

                if not model_choice:
                    selected_model = config.default_model
                    break

                if model_choice.isdigit():
                    model_idx = int(model_choice) - 1
                    model_names = list(models.keys())
                    if 0 <= model_idx < len(model_names):
                        selected_model = model_names[model_idx]
                        break

                print("❌ Please enter a number 1-5")

            except KeyboardInterrupt:
                print("\n\n👋 Cancelled")
                return

        print(f"\n🚀 Processing {len(selected_files)} file(s) with {selected_model} model...")
        print("💡 You can close the laptop lid safely during processing")
        print()

        # Process selected files with chosen model
        successful, failed = transcriber.process_files_sequential(selected_files, selected_model)

        print("\n🎉 Processing completed!")
        print(f"   ✅ Successful: {successful}")
        if failed:
            print(f"   ❌ Failed: {len(failed)}")
            for fail in failed:
                print(f"      - {fail}")

    else:
        print("📁 No media files found in input_media/")
        print("💡 Add files to input_media/ folder or choose from menu:")
        print()
        run_interactive_mode(transcriber)


def main():
    """Main application entry point."""
    try:
        # Create CLI handler and parse arguments
        cli = CLI()
        args = cli.parse_args()

        # Set up logging based on arguments
        cli.setup_logging_from_args(args)

        # Get configuration
        config = cli.get_config_from_args(args)

        # Handle daemon commands
        if args.daemon_action:
            success = cli.handle_daemon_command(args, config)
            sys.exit(0 if success else 1)

        # Smart auto-mode: program decides best approach
        if not args.batch_mode and not args.files:
            cli.display_welcome()
            transcriber = TranscriberCore(config)
            run_smart_mode(transcriber, config)
        else:
            # Specific files or batch mode - run normally
            transcriber = TranscriberCore(config)

            if args.files:
                # Process specific files
                files_to_process = [Path(f) for f in args.files if Path(f).exists()]
                if not files_to_process:
                    print("❌ No valid files specified")
                    sys.exit(1)

                print(f"📁 Processing {len(files_to_process)} specified file(s)")
                successful, failed = transcriber.process_files_sequential(
                    files_to_process, args.model
                )

            else:
                # Batch process all files in input directory
                files = transcriber.get_media_files()
                if not files:
                    print("❌ No media files found in input directory")
                    sys.exit(1)

                print(f"📁 Found {len(files)} file(s) in input directory")
                successful, failed = transcriber.process_files_sequential(
                    files, args.model
                )

            # Summary
            print("\n🎉 Batch processing completed!")
            print(f"   ✅ Successful: {successful}")
            if failed:
                print(f"   ❌ Failed: {len(failed)}")
                for fail in failed:
                    print(f"      - {fail}")

            sys.exit(0 if not failed else 1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
