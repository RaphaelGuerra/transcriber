#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced transcriber
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import torch

# Import modules to test
from config import TranscriberConfig
from file_handler import FileHandler
from transcriber_enhanced import (
    DeviceManager,
    EnhancedTranscriber,
    ProgressTracker,
    SubtitleGenerator,
    TranscriptionError,
    TranscriptionResult,
)


class TestDeviceManager(unittest.TestCase):
    """Test device management functionality."""
    
    def test_get_best_device_cpu(self):
        """Test device selection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            device = DeviceManager.get_best_device()
            self.assertEqual(device, "cpu")
    
    def test_get_best_device_gpu(self):
        """Test device selection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', return_value="NVIDIA GeForce RTX 3090"):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_props.return_value.total_memory = 24 * 1024**3  # 24 GB
                    device = DeviceManager.get_best_device()
                    self.assertEqual(device, "cuda")
    
    def test_get_device_info(self):
        """Test device information retrieval."""
        with patch('torch.cuda.is_available', return_value=False):
            info = DeviceManager.get_device_info()
            self.assertEqual(info["device"], "cpu")
            self.assertFalse(info["cuda_available"])
            self.assertIsNone(info["gpu_name"])


class TestProgressTracker(unittest.TestCase):
    """Test progress tracking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = ProgressTracker(total_files=5)
    
    def test_initial_state(self):
        """Test initial tracker state."""
        self.assertEqual(self.tracker.total_files, 5)
        self.assertEqual(self.tracker.processed, 0)
        self.assertEqual(self.tracker.successful, 0)
        self.assertEqual(self.tracker.failed, 0)
    
    def test_update_success(self):
        """Test updating with successful result."""
        result = TranscriptionResult(
            success=True,
            file_path=Path("test.mp4"),
            duration_seconds=10.5,
            character_count=1000
        )
        self.tracker.update(result, file_size_mb=50.0)
        
        self.assertEqual(self.tracker.processed, 1)
        self.assertEqual(self.tracker.successful, 1)
        self.assertEqual(self.tracker.failed, 0)
        self.assertEqual(len(self.tracker.file_times), 1)
        self.assertEqual(self.tracker.file_times[0], 10.5)
    
    def test_update_failure(self):
        """Test updating with failed result."""
        result = TranscriptionResult(
            success=False,
            file_path=Path("test.mp4"),
            error_message="Test error"
        )
        self.tracker.update(result, file_size_mb=50.0)
        
        self.assertEqual(self.tracker.processed, 1)
        self.assertEqual(self.tracker.successful, 0)
        self.assertEqual(self.tracker.failed, 1)
        self.assertEqual(len(self.tracker.file_times), 0)
    
    def test_get_stats(self):
        """Test statistics calculation."""
        # Add some results
        for i in range(3):
            result = TranscriptionResult(
                success=True,
                file_path=Path(f"test{i}.mp4"),
                duration_seconds=10.0 + i,
                character_count=1000
            )
            self.tracker.update(result, file_size_mb=50.0)
        
        stats = self.tracker.get_stats()
        
        self.assertEqual(stats["processed"], 3)
        self.assertEqual(stats["successful"], 3)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(stats["remaining"], 2)
        self.assertEqual(stats["success_rate"], 100.0)
        self.assertIn("avg_time_seconds", stats)
        self.assertIn("eta_seconds", stats)


class TestSubtitleGenerator(unittest.TestCase):
    """Test subtitle generation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SubtitleGenerator()
        self.test_segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "This is a test"},
            {"start": 5.0, "end": 7.5, "text": "Subtitle generation"},
        ]
    
    def test_format_timestamp_srt(self):
        """Test SRT timestamp formatting."""
        timestamp = self.generator.format_timestamp_srt(3661.5)  # 1h 1m 1.5s
        self.assertEqual(timestamp, "01:01:01,500")
    
    def test_format_timestamp_vtt(self):
        """Test WebVTT timestamp formatting."""
        timestamp = self.generator.format_timestamp_vtt(3661.5)  # 1h 1m 1.5s
        self.assertEqual(timestamp, "01:01:01.500")
    
    def test_generate_srt(self):
        """Test SRT file generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            success = self.generator.generate_srt(self.test_segments, temp_path)
            self.assertTrue(success)
            
            # Read and verify content
            content = temp_path.read_text()
            self.assertIn("1\n", content)
            self.assertIn("00:00:00,000 --> 00:00:02,500", content)
            self.assertIn("Hello world", content)
            self.assertIn("2\n", content)
            self.assertIn("This is a test", content)
        finally:
            temp_path.unlink()
    
    def test_generate_vtt(self):
        """Test WebVTT file generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vtt', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            success = self.generator.generate_vtt(self.test_segments, temp_path)
            self.assertTrue(success)
            
            # Read and verify content
            content = temp_path.read_text()
            self.assertIn("WEBVTT", content)
            self.assertIn("00:00:00.000 --> 00:00:02.500", content)
            self.assertIn("Hello world", content)
        finally:
            temp_path.unlink()
    
    def test_generate_json(self):
        """Test JSON file generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        test_result = {
            "text": "Full transcription text",
            "segments": self.test_segments,
            "language": "en"
        }
        
        try:
            success = self.generator.generate_json(test_result, temp_path)
            self.assertTrue(success)
            
            # Read and verify content
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(data["text"], "Full transcription text")
            self.assertEqual(len(data["segments"]), 3)
            self.assertEqual(data["language"], "en")
        finally:
            temp_path.unlink()


class TestFileHandler(unittest.TestCase):
    """Test file handling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input"
        self.output_dir = self.temp_dir / "output"
        self.temp_work_dir = self.temp_dir / "temp"
        
        self.file_handler = FileHandler(
            self.input_dir, self.output_dir, self.temp_work_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_directory_creation(self):
        """Test that directories are created."""
        self.assertTrue(self.input_dir.exists())
        self.assertTrue(self.output_dir.exists())
        self.assertTrue(self.temp_work_dir.exists())
    
    def test_get_media_files(self):
        """Test media file discovery."""
        # Create test files
        test_files = [
            self.input_dir / "video1.mp4",
            self.input_dir / "audio1.mp3",
            self.input_dir / "document.txt",  # Should be ignored
        ]
        
        for file_path in test_files:
            file_path.touch()
        
        # Get media files
        supported_extensions = [".mp4", ".mp3", ".wav", ".avi"]
        media_files = self.file_handler.get_media_files(supported_extensions)
        
        # Check results
        self.assertEqual(len(media_files), 2)
        media_names = [f.name for f in media_files]
        self.assertIn("video1.mp4", media_names)
        self.assertIn("audio1.mp3", media_names)
        self.assertNotIn("document.txt", media_names)
    
    def test_validate_file(self):
        """Test file validation."""
        # Create a test file
        test_file = self.input_dir / "test.mp4"
        test_file.write_text("test content")
        
        # Test valid file
        is_valid, message = self.file_handler.validate_file(test_file, max_size_gb=1.0)
        self.assertTrue(is_valid)
        self.assertEqual(message, "OK")
        
        # Test non-existent file
        is_valid, message = self.file_handler.validate_file(
            self.input_dir / "nonexistent.mp4", max_size_gb=1.0
        )
        self.assertFalse(is_valid)
        self.assertIn("does not exist", message)
    
    def test_save_transcription(self):
        """Test saving transcription with metadata."""
        output_path = self.output_dir / "test_transcription.txt"
        content = "This is the transcribed text."
        metadata = {
            "source_file": "test.mp4",
            "model": "base",
            "processing_time_seconds": 10.5
        }
        
        self.file_handler.save_transcription(content, output_path, metadata)
        
        # Verify file exists and content
        self.assertTrue(output_path.exists())
        saved_content = output_path.read_text()
        self.assertIn(content, saved_content)
        self.assertIn("source_file: test.mp4", saved_content)
        self.assertIn("model: base", saved_content)
    
    def test_estimate_duration(self):
        """Test duration estimation."""
        # Create a test file with known size
        test_file = self.input_dir / "test.mp4"
        test_file.write_bytes(b"0" * (10 * 1024 * 1024))  # 10 MB
        
        # Test video estimation (10 MB should be ~1 minute)
        duration = self.file_handler.estimate_duration(test_file, "video")
        self.assertAlmostEqual(duration, 1.0, places=1)
        
        # Test audio estimation (10 MB should be ~10 minutes)
        duration = self.file_handler.estimate_duration(test_file, "audio")
        self.assertAlmostEqual(duration, 10.0, places=1)


class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TranscriberConfig()
        
        self.assertEqual(config.default_model, "base")
        self.assertIn("tiny", config.available_models)
        self.assertIn(".mp4", config.supported_extensions)
        self.assertTrue(config.parallel_processing)
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = TranscriberConfig()
        
        # Test with valid configuration
        issues = config.validate()
        # Should have no issues if directories are writable
        # Note: May have issues in some test environments
        
        # Test with invalid model
        config.default_model = "invalid_model"
        issues = config.validate()
        self.assertTrue(any("Invalid default model" in issue for issue in issues))
    
    def test_supported_extensions(self):
        """Test getting supported extensions."""
        config = TranscriberConfig()
        
        # Get all extensions
        all_extensions = config.get_supported_extensions()
        self.assertIn(".mp4", all_extensions)
        self.assertIn(".mp3", all_extensions)
        
        # Get video extensions only
        video_extensions = config.get_supported_extensions("video")
        self.assertIn(".mp4", video_extensions)
        self.assertNotIn(".mp3", video_extensions)
        
        # Get audio extensions only
        audio_extensions = config.get_supported_extensions("audio")
        self.assertIn(".mp3", audio_extensions)
        self.assertNotIn(".mp4", audio_extensions)


class TestEnhancedTranscriber(unittest.TestCase):
    """Test enhanced transcriber functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = TranscriberConfig(
            input_dir=self.temp_dir / "input",
            output_dir=self.temp_dir / "output",
            temp_dir=self.temp_dir / "temp"
        )
        
        with patch('transcriber_enhanced.get_config', return_value=self.config):
            self.transcriber = EnhancedTranscriber(self.config)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('whisper.load_model')
    def test_load_model(self, mock_load_model):
        """Test model loading."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=False):
            model = self.transcriber.load_model("base")
        
        mock_load_model.assert_called_once_with("base", device="cpu")
        self.assertEqual(self.transcriber.model, mock_model)
        self.assertEqual(self.transcriber.model_name, "base")
        self.assertEqual(self.transcriber.device, "cpu")
    
    @patch('whisper.load_model')
    def test_load_model_gpu(self, mock_load_model):
        """Test model loading with GPU."""
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', return_value="Test GPU"):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_props.return_value.total_memory = 8 * 1024**3
                    model = self.transcriber.load_model("base", device="cuda")
        
        mock_load_model.assert_called_once_with("base", device="cuda")
        self.assertEqual(self.transcriber.device, "cuda")
    
    def test_check_memory(self):
        """Test memory checking."""
        # Create a small test file
        test_file = self.config.input_dir / "test.mp4"
        test_file.write_bytes(b"0" * 1024)  # 1 KB
        
        # Should pass for small file
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.available = 8 * 1024**3  # 8 GB available
            result = self.transcriber._check_memory(test_file)
            self.assertTrue(result)
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Set up mock model
        self.transcriber.model = MagicMock()
        self.transcriber.device = "cuda"
        
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            self.transcriber.cleanup()
        
        self.assertIsNone(self.transcriber.model)
        mock_empty_cache.assert_called_once()


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceManager))
    suite.addTests(loader.loadTestsFromTestCase(TestProgressTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestSubtitleGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestFileHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedTranscriber))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())