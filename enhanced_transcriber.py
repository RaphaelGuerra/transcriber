#!/usr/bin/env python3
"""
Enhanced Video/Audio Transcriber with Progress Tracking

This improved version includes:
- Progress bars and real-time status updates
- Organized folder structure (input_media, output_transcriptions)
- Multiple file management
- Time estimation and completion status
- Better user interface
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
import whisper
from tqdm import tqdm
import shutil
import argparse

class EnhancedTranscriber:
    def __init__(self):
        script_dir = Path(__file__).parent
        self.input_dir = script_dir / "input_media"
        self.output_dir = script_dir / "output_transcriptions"
        self.temp_dir = script_dir / "temp"
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
        self.whisper_model = None
        self.current_progress = 0
        self.is_processing = False
        
    def get_media_files(self):
        """Get all media files from input directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
        
        media_files = []
        
        # Add video files
        for ext in video_extensions:
            media_files.extend(self.input_dir.glob(f'*{ext}'))
        
        # Add audio files
        for ext in audio_extensions:
            media_files.extend(self.input_dir.glob(f'*{ext}'))
        
        return sorted(media_files)
    
    def display_file_info(self, media_files):
        """Display information about available media files."""
        if not media_files:
            print("❌ No media files found in input_media/ directory.")
            print("\nTo add files:")
            print("1. Place video/audio files in the 'input_media' folder")
            print("2. Supported formats: mp4, avi, mov, mkv, webm, flv, mp3, wav, m4a, aac, flac, ogg")
            return []
        
        print(f"🎬 Found {len(media_files)} media file(s) in input_media/")
        print("-" * 80)
        
        for i, media in enumerate(media_files, 1):
            size_mb = media.stat().st_size / (1024 * 1024)
            modified_time = datetime.fromtimestamp(media.stat().st_mtime)
            
            print(f"{i:2d}. {media.name}")
            print(f"    📁 Size: {size_mb:.1f} MB")
            print(f"    📅 Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Estimate duration based on file size (rough approximation)
            if media.suffix.lower() in ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']:
                # Audio files: roughly 1 MB per minute for MP3
                estimated_minutes = size_mb
                print(f"    ⏱️  Estimated duration: ~{estimated_minutes:.0f} minutes")
            else:
                # Video files: roughly 10-20 MB per minute
                estimated_minutes = size_mb / 15
                print(f"    ⏱️  Estimated duration: ~{estimated_minutes:.0f} minutes")
            
            print()
        
        return media_files
    
    def select_model(self):
        """Let user select Whisper model."""
        print("🎯 Available Whisper Models:")
        print("-" * 50)
        models = [
            ("tiny", "Fastest, least accurate (39M parameters)") ,
            ("base", "Good balance (74M parameters) - RECOMMENDED") ,
            ("small", "Better accuracy (244M parameters)") ,
            ("medium", "High accuracy (769M parameters)") ,
            ("large", "Best accuracy (1550M parameters)")
        ]
        
        for i, (model, description) in enumerate(models, 1):
            print(f"{i}. {model:6} - {description}")
        
        while True:
            try:
                choice = input(f"\n🎯 Choose model (1-5) [2]: ").strip()
                if not choice:
                    choice = "2"
                
                choice_num = int(choice)
                if 1 <= choice_num <= 5:
                    model_name = models[choice_num - 1][0]
                    print(f"✅ Selected: {model_name} model")
                    return model_name
                else:
                    print("❌ Please enter a number between 1-5")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def select_files(self, media_files):
        """Let user select which files to transcribe."""
        if not media_files:
            return []

        if len(media_files) == 1:
            print(f"\n🎬 Only one file found: {media_files[0].name}")
            response = input("Transcribe this file? (y/n) [y]: ").strip().lower()
            if response in ['', 'y', 'yes']:
                return [media_files[0]]
            else:
                return []
        
        print(f"\n🎬 Select files to transcribe:")
        print("You can enter a single number, a range of numbers (e.g., 1-3), or a mix of both separated by commas (e.g., 1,3-5,7).")
        print("You can also enter 'all' to transcribe all files, or 'q' to quit.")
        
        while True:
            choice = input(f"\nEnter your selection: ").strip().lower()
            
            if choice == 'q':
                return []
            elif choice == 'all':
                print(f"✅ Selected all {len(media_files)} files.")
                return media_files
            else:
                try:
                    selected_indices = set()
                    parts = choice.split(',')
                    for part in parts:
                        part = part.strip()
                        if '-' in part:
                            start, end = map(int, part.split('-'))
                            for i in range(start, end + 1):
                                selected_indices.add(i - 1)
                        else:
                            selected_indices.add(int(part) - 1)
                    
                    selected_files = []
                    for i in sorted(list(selected_indices)):
                        if 0 <= i < len(media_files):
                            selected_files.append(media_files[i])
                        else:
                            print(f"❌ Invalid file number: {i + 1}")

                    if selected_files:
                        print("\nYou have selected the following files:")
                        for i, file in enumerate(selected_files, 1):
                            print(f"  {i}. {file.name}")
                        
                        confirm = input("\nDo you want to transcribe these files? (y/n) [y]: ").strip().lower()
                        if confirm in ['', 'y', 'yes']:
                            return selected_files
                        else:
                            print("Selection cancelled.")
                            return []
                    else:
                        print("❌ No valid files selected. Please try again.")
                        
                except ValueError:
                    print("❌ Invalid input. Please enter numbers, ranges, or 'all'.")
    
    def estimate_transcription_time(self, file_path, model_name):
        """Estimate transcription time based on file size and model."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Base time estimates (in minutes) per MB for different models
        time_per_mb = {
            'tiny': 0.1,    # 6 seconds per MB
            'base': 0.2,    # 12 seconds per MB
            'small': 0.4,   # 24 seconds per MB
            'medium': 0.8,  # 48 seconds per MB
            'large': 1.2    # 72 seconds per MB
        }
        
        estimated_minutes = size_mb * time_per_mb.get(model_name, 0.2)
        return estimated_minutes
    
    def transcribe_with_progress(self, file_path, model_name):
        """Transcribe a file with real-time progress tracking."""
        print(f"\n🚀 Starting transcription of: {file_path.name}")
        print(f"   Model: {model_name}")
        print(f"   File size: {file_path.stat().st_size / (1024 * 1024):.1f} MB")
        
        # Estimate time
        estimated_minutes = self.estimate_transcription_time(file_path, model_name)
        print(f"   ⏱️  Estimated time: ~{estimated_minutes:.1f} minutes")
        print()
        
        start_time = time.time()
        
        try:
            # Load model with progress
            print("📥 Loading Whisper model...")
            model = whisper.load_model(model_name)
            print("✅ Model loaded successfully!")
            
            # Transcribe with progress
            print("🎬 Transcribing audio...")
            
            transcription_result = {}
def transcribe_target():
                result = model.transcribe(str(file_path), verbose=False)
                transcription_result['text'] = result['text']

transcription_thread = threading.Thread(target=transcribe_target)
transcription_thread.start()

with tqdm(total=int(estimated_minutes * 60), desc="Transcription Progress", unit="s") as pbar:
    while transcription_thread.is_alive():
        transcription_thread.join(1)
        pbar.update(1)
    pbar.n = pbar.total
    pbar.refresh()

transcription = transcription_result.get('text')

if transcription is None:
    raise Exception("Transcription failed.")

            # Calculate actual time taken
elapsed_time = time.time() - start_time
elapsed_minutes = elapsed_time / 60

print(f"\n✅ Transcription completed!")
print(f"   ⏱️  Time taken: {elapsed_minutes:.1f} minutes")
print(f"   📝 Characters: {len(transcription):,}")

            # Save transcription
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{file_path.stem}_{timestamp}.txt"
output_path = self.output_dir / output_filename

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(transcription)

print(f"   📄 Saved to: {output_path}")

return transcription, output_path
            
        except Exception as e:
            print(f"❌ Error transcribing {file_path.name}: {e}")
            return None, None
    
    def process_multiple_files(self, selected_files, model_name):
        """Process multiple files with progress tracking."""
        total_files = len(selected_files)
        completed_files = 0
        failed_files = []
        
        print(f"\n🎯 Processing {total_files} file(s) with {model_name} model...")
        print("=" * 80)
        
        for i, file_path in enumerate(selected_files, 1):
            print(f"\n📁 File {i}/{total_files}: {file_path.name}")
            print("-" * 60)
            
            transcription, output_path = self.transcribe_with_progress(file_path, model_name)
            
            if transcription:
                completed_files += 1
                print(f"✅ File {i}/{total_files} completed successfully!")
            else:
                failed_files.append(file_path.name)
                print(f"❌ File {i}/{total_files} failed!")
            
            # Show overall progress
            progress = (i / total_files) * 100
            print(f"📊 Overall progress: {progress:.1f}% ({i}/{total_files})")
            
            if i < total_files:
                print("\n⏳ Moving to next file...")
                time.sleep(2)  # Brief pause between files
        
        # Final summary
        print("\n" + "=" * 80)
        print("🎉 TRANSCRIPTION SESSION COMPLETED!")
        print("=" * 80)
        print(f"✅ Successfully processed: {completed_files}/{total_files} files")
        
        if failed_files:
            print(f"❌ Failed files: {len(failed_files)}")
            for failed in failed_files:
                print(f"   - {failed}")
        
        print(f"\n📁 Output folder: {self.output_dir}")
        
        print(f"\n📁 Input folder: {self.input_dir}")
        
        return completed_files, failed_files
    
    def show_output_files(self):
        """Show existing transcription files."""
        output_files = list(self.output_dir.glob("*.txt"))
        
        if not output_files:
            print("\n📁 No transcription files found in output_transcriptions/")
            return
        
        print(f"\n📁 Found {len(output_files)} transcription file(s):")
        print("-" * 60)
        
        for i, file_path in enumerate(output_files, 1):
            size_kb = file_path.stat().st_size / 1024
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            print(f"{i:2d}. {file_path.name}")
            print(f"    📄 Size: {size_kb:.1f} KB")
            print(f"    📅 Created: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    
    def run_test_mode(self):
        """Run the transcriber in a non-interactive test mode."""
        print("🔬 Running in test mode...")
        test_file = self.input_dir / "test_audio.mp3"
        if not test_file.exists():
            print("❌ Test audio file not found at 'input_media/test_audio.mp3'")
            return

        model_name = "tiny"
        selected_files = [test_file]

        self.process_multiple_files(selected_files, model_name)

    def run(self):
        """Main application loop."""
        while True:
            print("\n" + "=" * 80)
            print("🎬 Enhanced Video/Audio Transcriber")
            print("=" * 80)
            
            # Show available files
            media_files = self.get_media_files()
            if not media_files:
                print("\n💡 To get started:")
                print("1. Place media files in the 'input_media' folder")
                print("2. Run this script again")
                
                print("\n3. Or view existing transcriptions")
                print("4. Exit")
                
                choice = input("\nChoose option (1-4) [1]: ").strip()
                if choice == '4':
                    print("👋 Goodbye!")
                    break
                elif choice == '3':
                    self.show_output_files()
                continue
            
            # Display file information
            self.display_file_info(media_files)
            
            # Show menu
            print("📋 Menu Options:")
            print("1. Transcribe files")
            print("2. View existing transcriptions")
            print("3. Exit")
            
            menu_choice = input("\nChoose option (1-3): ").strip()
            
            if menu_choice == '1':
                # Select model and files
                model_name = self.select_model()
                selected_files = self.select_files(media_files)
                
                if selected_files:
                    self.process_multiple_files(selected_files, model_name)
                    
                    # Ask if user wants to continue
                    continue_choice = input("\n🔄 Process more files? (y/n) [n]: ").strip().lower()
                    if continue_choice not in ['y', 'yes']:
                        break
                
            elif menu_choice == '2':
                self.show_output_files()
                
            elif menu_choice == '3':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced Video/Audio Transcriber")
    parser.add_argument("--test", action="store_true", help="Run in non-interactive test mode.")
    args = parser.parse_args()

    try:
        transcriber = EnhancedTranscriber()
        if args.test:
            transcriber.run_test_mode()
        else:
            transcriber.run()
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
