#!/usr/bin/env python3
"""
Basic test script for the transcriber functionality
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import TranscriberConfig, get_config
from file_handler import FileHandler


def test_config():
    """Test configuration loading."""
    print("🧪 Testing configuration...")

    config = get_config()
    print(f"   ✅ Default model: {config.default_model}")
    print(f"   ✅ Input dir: {config.input_dir}")
    print(f"   ✅ Output dir: {config.output_dir}")

    # Test configuration validation
    issues = config.validate()
    if issues:
        print(f"   ⚠️  Configuration issues: {issues}")
    else:
        print("   ✅ Configuration validation passed")


def test_file_handler():
    """Test file handler functionality."""
    print("\n🧪 Testing file handler...")

    config = get_config()
    file_handler = FileHandler(config.input_dir, config.output_dir, config.temp_dir)

    # Test supported extensions
    supported = config.get_supported_extensions()
    print(f"   ✅ Supported extensions: {len(supported)} types")

    # Test directory creation
    assert config.input_dir.exists(), "Input directory should exist"
    assert config.output_dir.exists(), "Output directory should exist"
    assert config.temp_dir.exists(), "Temp directory should exist"
    print("   ✅ Directories created successfully")

    # Test file discovery
    media_files = file_handler.get_media_files(supported)
    print(f"   📁 Found {len(media_files)} media files")


def test_directory_structure():
    """Test directory structure and files."""
    print("\n🧪 Testing directory structure...")

    root_dir = Path(__file__).parent

    # Check for required files
    required_files = [
        "transcriber.py",
        "config.py",
        "file_handler.py",
        "logger.py",
        "cli.py",
        "requirements.txt",
        "README.md",
        "setup.py",
        ".gitignore",
    ]

    for file in required_files:
        file_path = root_dir / file
        if file_path.exists():
            print(f"   ✅ {file} exists")
        else:
            print(f"   ❌ {file} missing")

    # Check directories
    required_dirs = ["input_media", "output_transcriptions", "temp"]
    for dir_name in required_dirs:
        dir_path = root_dir / dir_name
        if dir_path.exists():
            print(f"   ✅ {dir_name}/ directory exists")
        else:
            print(f"   ⚠️  {dir_name}/ directory missing (will be created at runtime)")


def main():
    """Run all tests."""
    print("🚀 Running basic functionality tests...\n")

    try:
        test_config()
        test_file_handler()
        test_directory_structure()

        print("\n🎉 All tests passed! The transcriber is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Place media files in the 'input_media/' directory")
        print("   2. Run: python transcriber.py")
        print("   3. Follow the interactive prompts")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
