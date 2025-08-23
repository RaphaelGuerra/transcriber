#!/usr/bin/env python3
"""
Fast and Simple Video/Audio Transcriber

A streamlined transcriber that focuses on speed and simplicity:
- Efficient model loading and reuse
- Batch processing with multiprocessing
- Smart file sorting for optimal completion time
- Real progress tracking with tqdm
- Memory optimization
- Fully interactive interface
"""

import os
import sys
import time
import gc
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import whisper
from tqdm import tqdm

class FastTranscriber:
    def __init__(self):
        script_dir = Path(__file__).parent
        self.input_dir = script_dir / "input_media"
        self.output_dir = script_dir / "output_transcriptions"
        
        # Create directories if they don't exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.model_name = None
    
    def get_media_files(self):
        """Get all supported media files from input directory (optimized)."""
        extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', 
                     '.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}
        
        # Use glob for faster file discovery
        media_files = []
        for ext in extensions:
            media_files.extend(self.input_dir.glob(f'*{ext}'))
            media_files.extend(self.input_dir.glob(f'*{ext.upper()}'))
        
        # Sort by file size (smaller first for faster completion)
        media_files.sort(key=lambda x: x.stat().st_size)
        return media_files
    
    def load_model(self, model_name):
        """Load Whisper model (cached for reuse)."""
        if self.model is None or self.model_name != model_name:
            print(f"📥 Loading {model_name} model...")
            self.model = whisper.load_model(model_name)
            self.model_name = model_name
            print("✅ Model loaded successfully!")
            
            # Force garbage collection after model loading
            gc.collect()
        return self.model
    
    def estimate_duration(self, file_path):
        """Estimate audio duration based on file size."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Rough estimates: audio ~1MB/min, video ~10MB/min
        if file_path.suffix.lower() in {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}:
            return size_mb  # minutes
        else:
            return size_mb / 10  # minutes
    
    def transcribe_file(self, file_path, model_name):
        """Transcribe a single file efficiently."""
        print(f"\n🎬 Transcribing: {file_path.name}")
        
        # Load model (cached)
        model = self.load_model(model_name)
        
        # Estimate duration for progress context
        estimated_minutes = self.estimate_duration(file_path)
        print(f"   ⏱️  Estimated duration: ~{estimated_minutes:.1f} minutes")
        
        start_time = time.time()
        
        try:
            # Transcribe with progress
            print("   🎵 Processing audio...")
            result = model.transcribe(str(file_path), verbose=False)
            
            if not result or 'text' not in result:
                raise Exception("Transcription failed - no text returned")
            
            transcription = result['text'].strip()
            
            # Calculate actual time
            elapsed_time = time.time() - start_time
            elapsed_minutes = elapsed_time / 60
            
            print(f"   ✅ Completed in {elapsed_time:.1f} seconds")
            print(f"   📝 Characters: {len(transcription):,}")
            
            # Save transcription
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{file_path.stem}_{timestamp}.txt"
            output_path = self.output_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
            
            print(f"   📄 Saved: {output_filename}")
            
            # Force garbage collection after each file
            gc.collect()
            
            return True, output_path
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False, None
    
    def transcribe_file_worker(self, args):
        """Worker function for multiprocessing."""
        file_path, model_name = args
        try:
            # Load model in worker process
            model = whisper.load_model(model_name)
            result = model.transcribe(str(file_path), verbose=False)
            
            if result and 'text' in result:
                transcription = result['text'].strip()
                
                # Save transcription
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"{file_path.stem}_{timestamp}.txt"
                output_path = self.output_dir / output_filename
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                
                return True, file_path.name, output_path
            else:
                return False, file_path.name, None
                
        except Exception as e:
            return False, file_path.name, str(e)
    
    def process_files_parallel(self, files, model_name, max_workers=None):
        """Process multiple files in parallel for maximum speed."""
        if not files:
            return 0, []
        
        if max_workers is None:
            max_workers = min(mp.cpu_count(), len(files))
        
        print(f"\n🚀 Processing {len(files)} file(s) with {model_name} model")
        print(f"   Using {max_workers} parallel workers")
        print("=" * 60)
        
        # Prepare arguments for multiprocessing
        args = [(file_path, model_name) for file_path in files]
        
        successful = 0
        failed = []
        
        # Process files in parallel with progress bar
        with mp.Pool(processes=max_workers) as pool:
            results = list(tqdm(
                pool.imap(self.transcribe_file_worker, args),
                total=len(files),
                desc="Transcribing files",
                unit="file"
            ))
        
        # Process results
        for success, filename, output_path in results:
            if success:
                successful += 1
                print(f"✅ {filename} completed")
            else:
                failed.append(filename)
                print(f"❌ {filename} failed: {output_path}")
        
        # Summary
        print(f"\n🎉 Completed: {successful}/{len(files)} files")
        if failed:
            print(f"❌ Failed: {', '.join(failed)}")
        
        return successful, failed
    
    def process_files_sequential(self, files, model_name):
        """Process files sequentially (original method)."""
        if not files:
            return 0, []
        
        total_files = len(files)
        successful = 0
        failed = []
        
        print(f"\n🎯 Processing {total_files} file(s) with {model_name} model")
        print("=" * 60)
        
        for i, file_path in enumerate(files, 1):
            print(f"\n📁 File {i}/{total_files}")
            success, output_path = self.transcribe_file(file_path, model_name)
            
            if success:
                successful += 1
            else:
                failed.append(file_path.name)
            
            # Show progress
            progress = (i / total_files) * 100
            print(f"📊 Progress: {progress:.0f}% ({i}/{total_files})")
        
        # Summary
        print(f"\n🎉 Completed: {successful}/{total_files} files")
        if failed:
            print(f"❌ Failed: {', '.join(failed)}")
        
        return successful, failed
    
    def show_files(self):
        """Display available media files."""
        files = self.get_media_files()
        
        if not files:
            print("❌ No media files found in input_media/")
            print("\nSupported formats: mp4, avi, mov, mkv, webm, flv, mp3, wav, m4a, aac, flac, ogg")
            return []
        
        print(f"🎬 Found {len(files)} media file(s):")
        for i, file_path in enumerate(files, 1):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"{i:2d}. {file_path.name} ({size_mb:.1f} MB)")
        
        return files
    
    def select_files(self, files):
        """Simple file selection."""
        if len(files) == 1:
            print(f"\n🎬 Only one file found: {files[0].name}")
            return files
        
        print(f"\n🎬 Select files to transcribe:")
        print("Enter numbers separated by commas (e.g., 1,3,5) or 'all' for all files")
        
        while True:
            choice = input("Selection: ").strip().lower()
            
            if choice == 'all':
                return files
            
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(',')]
                selected = [files[i] for i in indices if 0 <= i < len(files)]
                
                if selected:
                    print(f"Selected: {len(selected)} file(s)")
                    return selected
                else:
                    print("❌ No valid files selected")
            except (ValueError, IndexError):
                print("❌ Invalid input. Use numbers like '1,3,5' or 'all'")
    
    def select_processing_mode(self, file_count):
        """Let user choose between parallel and sequential processing."""
        if file_count <= 1:
            return False
        
        print(f"\n🚀 Processing mode selection:")
        print(f"   Files to process: {file_count}")
        print(f"   Available CPU cores: {mp.cpu_count()}")
        
        if file_count > 1:
            print("\n   Parallel processing:")
            print("   ✅ Faster completion time")
            print("   ✅ Better CPU utilization")
            print("   ⚠️  Higher memory usage")
            print("   ⚠️  Less detailed progress per file")
            
            print("\n   Sequential processing:")
            print("   ✅ Lower memory usage")
            print("   ✅ Detailed progress per file")
            print("   ⚠️  Slower overall completion")
            
            while True:
                choice = input("\nUse parallel processing? (y/n) [y]: ").strip().lower()
                if choice in ['', 'y', 'yes']:
                    return True
                elif choice in ['n', 'no']:
                    return False
                else:
                    print("❌ Please enter 'y' or 'n'")
        
        return False
    
    def select_workers(self, file_count):
        """Let user choose number of parallel workers."""
        max_workers = min(mp.cpu_count(), file_count)
        
        print(f"\n🔧 Parallel processing configuration:")
        print(f"   Recommended workers: {max_workers}")
        print(f"   Available CPU cores: {mp.cpu_count()}")
        print(f"   Files to process: {file_count}")
        
        while True:
            try:
                choice = input(f"\nNumber of workers (1-{max_workers}) [{max_workers}]: ").strip()
                if not choice:
                    return max_workers
                
                workers = int(choice)
                if 1 <= workers <= max_workers:
                    return workers
                else:
                    print(f"❌ Please enter a number between 1 and {max_workers}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def show_outputs(self):
        """Display existing transcription files."""
        output_files = list(self.output_dir.glob("*.txt"))
        
        if not output_files:
            print("📁 No transcription files found")
            return
        
        print(f"\n📁 Found {len(output_files)} transcription file(s):")
        for file_path in sorted(output_files, key=lambda x: x.stat().st_mtime, reverse=True):
            size_kb = file_path.stat().st_size / 1024
            modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"   {file_path.name} ({size_kb:.1f} KB, {modified.strftime('%Y-%m-%d %H:%M')})")
    
    def run(self):
        """Main application loop."""
        print("🎬 Fast Video/Audio Transcriber")
        print("=" * 40)
        
        while True:
            files = self.show_files()
            
            if not files:
                print("\n💡 Place media files in 'input_media/' folder and run again")
                break
            
            print("\n📋 Options:")
            print("1. Transcribe files")
            print("2. View transcriptions")
            print("3. Exit")
            
            choice = input("\nChoice (1-3): ").strip()
            
            if choice == '1':
                # Model selection
                print("\n🎯 Model options:")
                models = ['tiny', 'base', 'small', 'medium', 'large']
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
                selected_files = self.select_files(files)
                if selected_files:
                    # Processing mode selection
                    parallel = self.select_processing_mode(len(selected_files))
                    
                    if parallel:
                        # Worker selection for parallel processing
                        max_workers = self.select_workers(len(selected_files))
                        self.process_files_parallel(selected_files, model_name, max_workers)
                    else:
                        # Sequential processing
                        self.process_files_sequential(selected_files, model_name)
                    
                    # Continue option
                    if input("\n🔄 Process more files? (y/n): ").strip().lower() not in ['y', 'yes']:
                        break
            
            elif choice == '2':
                self.show_outputs()
            
            elif choice == '3':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice")

def main():
    """Main function."""
    try:
        transcriber = FastTranscriber()
        transcriber.run()
    except KeyboardInterrupt:
        print("\n\n⏹️  Cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()