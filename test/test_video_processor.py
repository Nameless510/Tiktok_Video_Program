#!/usr/bin/env python3
"""
Test video processor functionality
- Video metadata extraction
- Keyframe extraction
"""

import os
import sys
import unittest
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_path = current_dir.parent / 'src'
sys.path.insert(0, str(src_path))

from video_processor import VideoProcessor

class TestVideoProcessor(unittest.TestCase):
    """Test video processor functionality"""
    
    def setUp(self):
        """Test preparation"""
        self.processor = VideoProcessor()
        self.test_video_path = "data/tiktok_videos/Download (7).mp4"  # Use existing test video
        self.test_output_dir = "test/test_output/keyframes"
        
        # Ensure output directory exists
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def test_video_metadata_extraction(self):
        """Test video metadata extraction"""
        print("\n=== Test Video Metadata Extraction ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            print("Please ensure there are test video files in the data/tiktok_videos directory")
            return
        
        # Extract metadata
        metadata = self.processor.get_video_metadata(self.test_video_path)
        
        # Validate metadata
        self.assertIsInstance(metadata, dict)
        self.assertIn('duration', metadata)
        self.assertIn('width', metadata)
        self.assertIn('height', metadata)
        self.assertIn('frame_rate', metadata)
        self.assertIn('file_size', metadata)
        
        print(f"Video metadata extraction successful:")
        print(f"  Duration: {metadata['duration']:.2f} seconds")
        print(f"  Resolution: {metadata['width']}x{metadata['height']}")
        print(f"  Frame rate: {metadata['frame_rate']:.2f} fps")
        print(f"  File size: {metadata['file_size']} bytes")
        
        # Validate data reasonableness
        self.assertGreater(metadata['duration'], 0)
        self.assertGreater(metadata['width'], 0)
        self.assertGreater(metadata['height'], 0)
        self.assertGreater(metadata['file_size'], 0)
    
    def test_keyframe_extraction(self):
        """Test keyframe extraction"""
        print("\n=== Test Keyframe Extraction ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            print("Please ensure there are test video files in the data/tiktok_videos directory")
            return
        
        # Extract keyframes
        output_dir, frame_count = self.processor.extract_keyframes(
            self.test_video_path, 
            self.test_output_dir
        )
        
        # Validate results
        self.assertIsInstance(output_dir, str)
        self.assertIsInstance(frame_count, int)
        self.assertGreaterEqual(frame_count, 0)
        
        print(f"Keyframe extraction successful:")
        print(f"  Output directory: {output_dir}")
        print(f"  Extracted frames: {frame_count}")
        
        # Check actual files
        if frame_count > 0:
            video_name = os.path.splitext(os.path.basename(self.test_video_path))[0]
            expected_files = [f for f in os.listdir(output_dir) 
                            if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')]
            print(f"  Actual file count: {len(expected_files)}")
            
            # Validate file existence
            self.assertEqual(len(expected_files), frame_count)
            
            # Check file sizes
            for file in expected_files[:3]:  # Only check first 3 files
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"    {file}: {file_size} bytes")
                self.assertGreater(file_size, 0)
    
    def test_invalid_video_path(self):
        """Test invalid video path handling"""
        print("\n=== Test Invalid Video Path Handling ===")
        
        # Test non-existent file
        invalid_path = "data/tiktok_videos/nonexistent.mp4"
        metadata = self.processor.get_video_metadata(invalid_path)
        
        # Should return default values
        self.assertEqual(metadata['duration'], 0)
        self.assertEqual(metadata['width'], 0)
        self.assertEqual(metadata['height'], 0)
        self.assertEqual(metadata['frame_rate'], 0)
        self.assertEqual(metadata['file_size'], 0)
        
        print("Invalid video path handling correct")

def run_tests():
    """Run all tests"""
    print("Starting video processor functionality tests...")
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestVideoProcessor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Output test results
    print(f"\nTest Results:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 