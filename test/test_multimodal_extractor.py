#!/usr/bin/env python3
"""
Test multimodal extractor functionality
- YOLO object detection
- BLIP image description
- OCR text recognition
- Qwen-VL multimodal analysis
"""

import os
import sys
import unittest
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_path = current_dir.parent / 'src'
sys.path.insert(0, str(src_path))

from multimodal_extractor import MultimodalExtractor
from video_processor import VideoProcessor

class TestMultimodalExtractor(unittest.TestCase):
    """Test multimodal extractor functionality"""
    
    def setUp(self):
        """Test preparation"""
        self.extractor = MultimodalExtractor()
        self.video_processor = VideoProcessor()
        self.test_video_path = "data/tiktok_videos/Download (7).mp4"
        self.test_output_dir = "test/test_output/multimodal"
        self.test_keyframes_dir = os.path.join(self.test_output_dir, "keyframes")
        
        # Ensure output directory exists
        os.makedirs(self.test_output_dir, exist_ok=True)
        os.makedirs(self.test_keyframes_dir, exist_ok=True)
    
    def test_yolo_analysis(self):
        """Test YOLO object detection analysis"""
        print("\n=== Test YOLO Object Detection Analysis ===")
        
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
        
        # Get first few keyframes for testing
        frame_files = [f for f in os.listdir(self.test_keyframes_dir) 
                      if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')][:2]
        
        if not frame_files:
            print("No keyframe files found")
            return
        
        # Test YOLO analysis
        for frame_file in frame_files:
            frame_path = os.path.join(self.test_keyframes_dir, frame_file)
            
            # Perform YOLO analysis
            yolo_results = self.extractor.analyze_with_yolo(frame_path)
            
            # Validate results
            self.assertIsInstance(yolo_results, dict)
            self.assertIn('objects', yolo_results)
            self.assertIn('summary', yolo_results)
            
            print(f"  {frame_file}:")
            print(f"    Number of detected objects: {len(yolo_results['objects'])}")
            print(f"    Summary: {yolo_results['summary']}")
            
            # Check object info
            for obj in yolo_results['objects'][:3]:  # Only show first 3
                self.assertIn('class', obj)
                self.assertIn('confidence', obj)
                self.assertIn('bbox', obj)
                print(f"      - {obj['class']} (confidence: {obj['confidence']:.3f})")
    
    def test_blip_analysis(self):
        """Test BLIP image description analysis"""
        print("\n=== Test BLIP Image Description Analysis ===")
        
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
            print("Keyframe extraction failed, skipping BLIP test")
            return
        
        # Get first few keyframes for testing
        frame_files = [f for f in os.listdir(self.test_keyframes_dir) 
                      if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')][:2]
        
        if not frame_files:
            print("No keyframe files found")
            return
        
        # Test BLIP analysis
        for frame_file in frame_files:
            frame_path = os.path.join(self.test_keyframes_dir, frame_file)
            
            # Perform BLIP analysis
            blip_results = self.extractor.analyze_with_blip(frame_path)
            
            # Validate results
            self.assertIsInstance(blip_results, dict)
            self.assertIn('description', blip_results)
            self.assertIn('confidence', blip_results)
            
            print(f"  {frame_file}:")
            print(f"    Description: {blip_results['description']}")
            print(f"    Confidence: {blip_results['confidence']:.3f}")
    
    def test_ocr_analysis(self):
        """Test OCR text recognition analysis"""
        print("\n=== Test OCR Text Recognition Analysis ===")
        
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
            print("Keyframe extraction failed, skipping OCR test")
            return
        
        # Get first few keyframes for testing
        frame_files = [f for f in os.listdir(self.test_keyframes_dir) 
                      if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')][:2]
        
        if not frame_files:
            print("No keyframe files found")
            return
        
        # Test OCR analysis
        for frame_file in frame_files:
            frame_path = os.path.join(self.test_keyframes_dir, frame_file)
            
            # Perform OCR analysis
            ocr_results = self.extractor.analyze_with_ocr(frame_path)
            
            # Validate results
            self.assertIsInstance(ocr_results, dict)
            self.assertIn('text', ocr_results)
            self.assertIn('confidence', ocr_results)
            
            print(f"  {frame_file}:")
            print(f"    Recognized text: {ocr_results['text']}")
            print(f"    Confidence: {ocr_results['confidence']:.3f}")
    
    def test_qwen_analysis(self):
        """Test Qwen-VL multimodal analysis"""
        print("\n=== Test Qwen-VL Multimodal Analysis ===")
        
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
            print("Keyframe extraction failed, skipping Qwen test")
            return
        
        # Get first few keyframes for testing
        frame_files = [f for f in os.listdir(self.test_keyframes_dir) 
                      if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')][:1]  # Only test 1, Qwen is slow
        
        if not frame_files:
            print("No keyframe files found")
            return
        
        # Test Qwen analysis
        frame_path = os.path.join(self.test_keyframes_dir, frame_files[0])
        
        # Simulate OCR and audio data
        ocr_results = {"text": "Test text", "confidence": 0.8}
        audio_transcript = "This is a test audio transcript"
        
        # Perform Qwen analysis
        qwen_results = self.extractor.analyze_with_qwen(
            frame_path, 
            ocr_results, 
            audio_transcript
        )
        
        # Validate results
        self.assertIsInstance(qwen_results, dict)
        self.assertIn('description', qwen_results)
        self.assertIn('categories', qwen_results)
        
        print(f"  {frame_files[0]}:")
        print(f"    Description: {qwen_results['description']}")
        print(f"    Categories: {qwen_results['categories']}")
    
    def test_complete_multimodal_analysis(self):
        """Test complete multimodal analysis process"""
        print("\n=== Test Complete Multimodal Analysis Process ===")
        
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
            print("Keyframe extraction failed, skipping complete analysis test")
            return
        
        # Get first few keyframes for testing
        frame_files = [f for f in os.listdir(self.test_keyframes_dir) 
                      if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')][:1]
        
        if not frame_files:
            print("No keyframe files found")
            return
        
        # Perform complete analysis
        frame_path = os.path.join(self.test_keyframes_dir, frame_files[0])
        
        # Simulate OCR and audio data
        ocr_results = {"text": "Test text", "confidence": 0.8}
        audio_transcript = "This is a test audio transcript"
        
        # Perform complete analysis
        results = self.extractor.analyze_frame_complete(
            frame_path, 
            ocr_results, 
            audio_transcript
        )
        
        # Validate results
        self.assertIsInstance(results, dict)
        self.assertIn('yolo', results)
        self.assertIn('blip', results)
        self.assertIn('ocr', results)
        self.assertIn('qwen', results)
        
        print(f"Complete multimodal analysis results:")
        print(f"  Number of YOLO objects: {len(results['yolo']['objects'])}")
        print(f"  BLIP description: {results['blip']['description']}")
        print(f"  OCR text: {results['ocr']['text']}")
        print(f"  Qwen description: {results['qwen']['description']}")
    
    def test_feature_extraction_from_directory(self):
        """Test feature extraction from directory"""
        print("\n=== Test Feature Extraction from Directory ===")
        
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
            print("Keyframe extraction failed, skipping directory feature extraction test")
            return
        
        # Simulate OCR and audio data
        ocr_results = {
            f"{video_name}_keyframe_0001.jpg": "Test text 1",
            f"{video_name}_keyframe_0002.jpg": "Test text 2"
        }
        audio_transcript = "This is a test audio transcript"
        
        # Perform feature extraction
        features = self.extractor.extract_qwen_features(
            self.test_keyframes_dir,
            video_name,
            ocr_results,
            audio_transcript
        )
        
        # Validate results
        self.assertIsInstance(features, dict)
        self.assertIn('frames', features)
        self.assertIn('summary', features)
        
        print(f"Directory feature extraction results:")
        print(f"   Processed frames: {len(features['frames'])}")
        print(f"   Summary: {features['summary']}")
    
    def test_csv_generation(self):
        """Test CSV generation functionality"""
        print("\n=== Test CSV Generation Functionality ===")
        
        # Create test data
        test_results = {
            'frames': [
                {
                    'frame': 'test1.jpg',
                    'description': 'Test description 1',
                    'categories': ['Beauty', 'Makeup', 'Lipstick']
                },
                {
                    'frame': 'test2.jpg', 
                    'description': 'Test description 2',
                    'categories': ['Fashion', 'Clothing', 'Dress']
                }
            ],
            'summary': 'This is a test video summary'
        }
        
        # Generate CSV
        csv_path = os.path.join(self.test_output_dir, "test_results.csv")
        success = self.extractor.generate_video_summary_csv(test_results, csv_path)
        
        # Validate results
        self.assertTrue(success)
        self.assertTrue(os.path.exists(csv_path))
        
        # Check file size
        file_size = os.path.getsize(csv_path)
        print(f"CSV generation successful:")
        print(f"   File path: {csv_path}")
        print(f"   File size: {file_size} bytes")
        self.assertGreater(file_size, 0)

def run_tests():
    """Run all tests"""
    print("Starting multimodal extractor functionality tests...")
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMultimodalExtractor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Output test results
    print(f"\nTest results:")
    print(f"   Run tests: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 