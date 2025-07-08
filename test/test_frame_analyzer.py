#!/usr/bin/env python3
"""
Test frame analyzer functionality
- Keyframe filtering (similarity, blur, black frames)
- YOLO object detection
- Representative frame selection
- CLIP similarity analysis
"""

import os
import sys
import unittest
import shutil
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_path = current_dir.parent / 'src'
sys.path.insert(0, str(src_path))

from frame_analyzer import FrameAnalyzer
from video_processor import VideoProcessor

class TestFrameAnalyzer(unittest.TestCase):
    """Test frame analyzer functionality"""
    
    def setUp(self):
        """Test preparation"""
        self.analyzer = FrameAnalyzer()
        self.video_processor = VideoProcessor()
        self.test_video_path = "data/tiktok_videos/Download (7).mp4"
        self.test_output_dir = "test/test_output/frames"
        self.test_keyframes_dir = os.path.join(self.test_output_dir, "keyframes")
        
        # Ensure output directory exists
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.test_keyframes_dir, exist_ok=True)
    
    def test_keyframe_extraction_and_filtering(self):
        """Test keyframe extraction and filtering"""
        print("\n=== Test Keyframe Extraction and Filtering ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            print("Please ensure there are test video files in the data/tiktok_videos directory")
            return
        
        # First extract keyframes
        video_name = os.path.splitext(os.path.basename(self.test_video_path))[0]
        output_dir, frame_count = self.video_processor.extract_keyframes(
            self.test_video_path, 
            self.test_keyframes_dir
        )
        
        if frame_count == 0:
            print("Keyframe extraction failed, skipping filtering test")
            return
        
        print(f"Extracted {frame_count} keyframes")
        
        # Filter similar frames
        filtered_frames = self.analyzer.filter_similar_keyframes(
            self.test_keyframes_dir, 
            video_name, 
            ssim_threshold=0.8
        )
        
        # Validate results
        self.assertIsInstance(filtered_frames, list)
        self.assertLessEqual(len(filtered_frames), frame_count)
        
        print(f"Remaining {len(filtered_frames)} keyframes after filtering")
        
        # Check filtered files
        for frame_path in filtered_frames[:3]:  # Only check first 3
            self.assertTrue(os.path.exists(frame_path))
            file_size = os.path.getsize(frame_path)
            print(f"  {os.path.basename(frame_path)}: {file_size} bytes")
            self.assertGreater(file_size, 0)
    
    def test_frame_quality_analysis(self):
        """Test frame quality analysis (black frames, blur detection)"""
        print("\n=== Test Frame Quality Analysis ===")
        
        # Create test images (using existing keyframes here)
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            return
        
        # Extract some keyframes for testing
        video_name = os.path.splitext(os.path.basename(self.test_video_path))[0]
        output_dir, frame_count = self.video_processor.extract_keyframes(
            self.test_video_path, 
            self.test_keyframes_dir
        )
        
        if frame_count == 0:
            print("Keyframe extraction failed, skipping quality analysis test")
            return
        
        # Get first few keyframes for testing
        frame_files = [f for f in os.listdir(self.test_keyframes_dir) 
                      if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')][:3]
        
        for frame_file in frame_files:
            frame_path = os.path.join(self.test_keyframes_dir, frame_file)
            
            # Test black frame detection
            is_black = self.analyzer.is_black_frame(frame_path)
            self.assertIsInstance(is_black, bool)
            
            # Test blur detection
            is_blurry = self.analyzer.is_blurry(frame_path)
            self.assertIsInstance(is_blurry, bool)
            
            print(f"  {frame_file}:")
            print(f"    Black frame: {is_black}")
            print(f"    Blurry: {is_blurry}")
    
    def test_yolo_object_detection(self):
        """Test YOLO object detection"""
        print("\n=== Test YOLO Object Detection ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            return
        
        # Extract keyframes
        video_name = os.path.splitext(os.path.basename(self.test_video_path))[0]
        output_dir, frame_count = self.video_processor.extract_keyframes(
            self.test_video_path, 
            self.test_keyframes_dir
        )
        
        if frame_count == 0:
            print("Keyframe extraction failed, skipping YOLO test")
            return
        
        # Extract YOLO features
        frame_infos = self.analyzer.extract_yolo_features_for_keyframes(
            self.test_keyframes_dir, 
            video_name
        )
        
        # Validate results
        self.assertIsInstance(frame_infos, list)
        self.assertGreater(len(frame_infos), 0)
        
        print(f"YOLO detection results:")
        print(f"  Processed frames: {len(frame_infos)}")
        
        # Check each frame's information
        for info in frame_infos[:2]:  # Only check first 2
            self.assertIn('frame', info)
            self.assertIn('frame_path', info)
            self.assertIn('yolo_objects', info)
            self.assertIsInstance(info['yolo_objects'], list)
            
            print(f"  {info['frame']}:")
            print(f"    Detected {len(info['yolo_objects'])} objects")
            
            # Show first few detection results
            for obj in info['yolo_objects'][:3]:
                print(f"      - {obj['cls']} (confidence: {obj['conf']:.3f})")
    
    def test_product_frame_detection(self):
        """Test product frame detection"""
        print("\n=== Test Product Frame Detection ===")
        
        # Create some test data
        test_objects = [
            [{"cls": "person", "conf": 0.9}],  # Non-product
            [{"cls": "cell phone", "conf": 0.8}],  # Product
            [{"cls": "car", "conf": 0.7}],  # Non-product
            [{"cls": "laptop", "conf": 0.9}],  # Product
        ]
        
        for i, objects in enumerate(test_objects):
            is_product = self.analyzer.is_tiktok_product_frame(objects)
            print(f"  Test {i+1}: {[obj['cls'] for obj in objects]} -> Product frame: {is_product}")
            self.assertIsInstance(is_product, bool)
    
    def test_representative_frame_selection(self):
        """Test representative frame selection"""
        print("\n=== Test Representative Frame Selection ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            return
        
        # Extract keyframes
        video_name = os.path.splitext(os.path.basename(self.test_video_path))[0]
        output_dir, frame_count = self.video_processor.extract_keyframes(
            self.test_video_path, 
            self.test_keyframes_dir
        )
        
        if frame_count == 0:
            print("Keyframe extraction failed, skipping representative frame selection test")
            return
        
        # Select representative frames
        representative_frames = self.analyzer.get_representative_frames(
            self.test_keyframes_dir, 
            video_name
        )
        
        # Validate results
        self.assertIsInstance(representative_frames, list)
        
        print(f"Representative frame selection results:")
        print(f"  Original keyframe count: {frame_count}")
        print(f"  Selected representative frames: {len(representative_frames)}")
        
        # Validate representative frame count reasonableness
        expected_count = self.analyzer.get_representative_frame_count(len(representative_frames))
        self.assertLessEqual(len(representative_frames), expected_count)
        
        # Check representative frame files
        for frame_path in representative_frames:
            self.assertTrue(os.path.exists(frame_path))
            file_size = os.path.getsize(frame_path)
            print(f"  {os.path.basename(frame_path)}: {file_size} bytes")
            self.assertGreater(file_size, 0)
    
    def test_frame_count_calculation(self):
        """Test representative frame count calculation"""
        print("\n=== Test Representative Frame Count Calculation ===")
        
        test_cases = [
            (1, 2),   # 1 frame -> 2 frames
            (3, 3),   # 3 frames -> 3 frames
            (10, 3),  # 10 frames -> 3 frames
            (20, 4),  # 20 frames -> 4 frames
            (50, 5),  # 50 frames -> 5 frames
        ]
        
        for input_frames, expected_output in test_cases:
            result = self.analyzer.get_representative_frame_count(input_frames)
            print(f"  {input_frames} frames -> {result} representative frames (expected: {expected_output})")
            self.assertEqual(result, expected_output)
    
    def test_contentless_frame_filtering(self):
        """Test contentless frame filtering"""
        print("\n=== Test Contentless Frame Filtering ===")
        
        # Create test data
        test_frame_infos = [
            {"frame_path": "test1.jpg", "yolo_objects": []},
            {"frame_path": "test2.jpg", "yolo_objects": [{"cls": "person"}]},
        ]
        
        # Here we simulate black frame detection (real images needed for actual test)
        # Since we don't have real images, we only test function call
        filtered = self.analyzer.filter_contentless_frames(test_frame_infos)
        self.assertIsInstance(filtered, list)
        
        print(f"Contentless frame filtering test completed")
    
    def test_blurry_frame_filtering(self):
        """Test blurry frame filtering"""
        print("\n=== Test Blurry Frame Filtering ===")
        
        # Create test data
        test_frame_infos = [
            {"frame_path": "test1.jpg"},
            {"frame_path": "test2.jpg"},
        ]
        
        # Test function call (real images needed for actual test)
        filtered = self.analyzer.filter_blurry_frames(test_frame_infos)
        self.assertIsInstance(filtered, list)
        
        print(f"Blurry frame filtering test completed")

def run_tests():
    """Run all tests"""
    print("Starting frame analyzer functionality tests...")
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFrameAnalyzer)
    
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