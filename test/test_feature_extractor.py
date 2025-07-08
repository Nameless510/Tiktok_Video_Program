#!/usr/bin/env python3
"""
Test feature extractor functionality
- Complete video feature extraction workflow
- Single video processing
- Batch video processing
- Result saving and validation
"""

import os
import sys
import unittest
import pandas as pd
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_path = current_dir.parent / 'src'
sys.path.insert(0, str(src_path))

from tiktok_feature_extractor import TikTokFeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    """Test feature extractor functionality"""
    
    def setUp(self):
        """Test preparation"""
        self.extractor = TikTokFeatureExtractor()
        self.test_video_path = "data/tiktok_videos/Download (7).mp4"
        self.test_output_dir = "test/test_output/features"
        self.test_csv_path = os.path.join(self.test_output_dir, "test_results.csv")
        
        # Ensure output directory exists
        os.makedirs(self.test_output_dir, exist_ok=True)
    
    def test_single_video_processing(self):
        """Test single video processing"""
        print("\n=== Test Single Video Processing ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            print("Please ensure there are test video files in the data/tiktok_videos directory")
            return
        
        # Process single video
        df = self.extractor.extract_features_from_single_video(
            video_path=self.test_video_path,
            output_folder=self.test_output_dir,
            csv_output_path=self.test_csv_path
        )
        
        # Validate results
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        print(f"Single video processing result:")
        print(f"  Number of videos processed: {len(df)}")
        print(f"  Number of columns: {len(df.columns)}")
        
        # Check for key columns
        expected_columns = [
            'video_name', 'video_path', 'duration', 'width', 'height', 
            'frame_rate', 'file_size', 'has_speech', 'speech_text', 
            'speech_text_length', 'keyframe_count', 'representative_frame_count'
        ]
        
        for col in expected_columns:
            if col in df.columns:
                print(f"  ✓ {col}: {df.iloc[0][col]}")
            else:
                print(f"  ✗ {col}: missing")
        
        # Validate data types
        self.assertIsInstance(df.iloc[0]['video_name'], str)
        self.assertIsInstance(df.iloc[0]['duration'], (int, float))
        self.assertIsInstance(df.iloc[0]['has_speech'], bool)
    
    def test_batch_video_processing(self):
        """Test batch video processing"""
        print("\n=== Test Batch Video Processing ===")
        
        # Check video directory
        video_folder = "data/tiktok_videos"
        if not os.path.exists(video_folder):
            print(f"Video directory does not exist: {video_folder}")
            return
        
        # Get video file list
        video_files = [f for f in os.listdir(video_folder) if f.lower().endswith('.mp4')]
        if len(video_files) == 0:
            print("No MP4 files found in video directory")
            return
        
        print(f"Found {len(video_files)} video files")
        
        # Only process first 2 videos for testing
        test_videos = video_files[:2]
        print(f"Testing first {len(test_videos)} videos")
        
        # Create temp directory for testing
        test_batch_dir = os.path.join(self.test_output_dir, "batch_test")
        os.makedirs(test_batch_dir, exist_ok=True)
        
        # Copy test videos to temp directory
        import shutil
        for video_file in test_videos:
            src_path = os.path.join(video_folder, video_file)
            dst_path = os.path.join(test_batch_dir, video_file)
            shutil.copy2(src_path, dst_path)
        
        # Batch processing
        df = self.extractor.extract_features_from_folder(
            video_folder=test_batch_dir,
            output_folder=self.test_output_dir,
            csv_output_path=os.path.join(self.test_output_dir, "batch_results.csv")
        )
        
        # Validate results
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertLessEqual(len(df), len(test_videos))
        
        print(f"Batch processing result:")
        print(f"  Number of videos processed: {len(df)}")
        print(f"  Number of columns: {len(df.columns)}")
        
        # Check each video's processing result
        for idx, row in df.iterrows():
            print(f"  Video {idx+1}: {row['video_name']}")
            print(f"    Duration: {row['duration']:.2f} seconds")
            print(f"    Resolution: {row['width']}x{row['height']}")
            print(f"    Has speech: {row['has_speech']}")
            print(f"    Keyframe count: {row['keyframe_count']}")
    
    def test_video_metadata_extraction(self):
        """Test video metadata extraction"""
        print("\n=== Test Video Metadata Extraction ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            return
        
        # Extract metadata
        metadata = self.extractor.extract_video_metadata(self.test_video_path)
        
        # Validate results
        self.assertIsInstance(metadata, dict)
        self.assertIn('duration', metadata)
        self.assertIn('width', metadata)
        self.assertIn('height', metadata)
        self.assertIn('frame_rate', metadata)
        self.assertIn('file_size', metadata)
        
        print(f"Video metadata:")
        print(f"  Duration: {metadata['duration']:.2f} seconds")
        print(f"  Resolution: {metadata['width']}x{metadata['height']}")
        print(f"  Frame rate: {metadata['frame_rate']:.2f} fps")
        print(f"  File size: {metadata['file_size']} bytes")
        
        # Validate data reasonableness
        self.assertGreater(metadata['duration'], 0)
        self.assertGreater(metadata['width'], 0)
        self.assertGreater(metadata['height'], 0)
        self.assertGreater(metadata['file_size'], 0)
    
    def test_audio_processing(self):
        """Test audio processing"""
        print("\n=== Test Audio Processing ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            return
        
        # Process audio
        audio_results = self.extractor.process_audio(self.test_video_path, self.test_output_dir)
        
        # Validate results
        self.assertIsInstance(audio_results, dict)
        self.assertIn('has_speech', audio_results)
        self.assertIn('speech_text', audio_results)
        self.assertIn('speech_text_length', audio_results)
        
        print(f"Audio processing result:")
        print(f"  Has speech: {audio_results['has_speech']}")
        print(f"  Transcribed text: {audio_results['speech_text'][:100]}{'...' if len(audio_results['speech_text']) > 100 else ''}")
        print(f"  Text length: {audio_results['speech_text_length']} characters")
        
        # Validate logical consistency
        if audio_results['has_speech']:
            self.assertGreater(len(audio_results['speech_text']), 0)
            self.assertEqual(len(audio_results['speech_text']), audio_results['speech_text_length'])
        else:
            self.assertEqual(len(audio_results['speech_text']), 0)
            self.assertEqual(audio_results['speech_text_length'], 0)
    
    def test_frame_processing(self):
        """Test frame processing"""
        print("\n=== Test Frame Processing ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            return
        
        # Process frames
        frame_results = self.extractor.process_frames(self.test_video_path, self.test_output_dir)
        
        # Validate results
        self.assertIsInstance(frame_results, dict)
        self.assertIn('keyframe_count', frame_results)
        self.assertIn('representative_frame_count', frame_results)
        self.assertIn('representative_frames', frame_results)
        
        print(f"Frame processing result:")
        print(f"  Keyframe count: {frame_results['keyframe_count']}")
        print(f"  Representative frame count: {frame_results['representative_frame_count']}")
        print(f"  Representative frame list: {[os.path.basename(f) for f in frame_results['representative_frames']]}")
        
        # Validate data reasonableness
        self.assertGreaterEqual(frame_results['keyframe_count'], 0)
        self.assertGreaterEqual(frame_results['representative_frame_count'], 0)
        self.assertLessEqual(frame_results['representative_frame_count'], frame_results['keyframe_count'])
        
        # Check representative frame files
        for frame_path in frame_results['representative_frames']:
            if os.path.exists(frame_path):
                file_size = os.path.getsize(frame_path)
                print(f"    {os.path.basename(frame_path)}: {file_size} bytes")
                self.assertGreater(file_size, 0)
    
    def test_multimodal_analysis(self):
        """Test multimodal analysis"""
        print("\n=== Test Multimodal Analysis ===")
        
        if not os.path.exists(self.test_video_path):
            print(f"Test video file does not exist: {self.test_video_path}")
            return
        
        # First process frames
        frame_results = self.extractor.process_frames(self.test_video_path, self.test_output_dir)
        
        if frame_results['representative_frame_count'] == 0:
            print("No representative frames, skipping multimodal analysis test")
            return
        
        # Process audio
        audio_results = self.extractor.process_audio(self.test_video_path, self.test_output_dir)
        
        # Multimodal analysis
        multimodal_results = self.extractor.process_multimodal_analysis(
            frame_results['representative_frames'],
            audio_results['speech_text'],
            self.test_output_dir
        )
        
        # Validate results
        self.assertIsInstance(multimodal_results, dict)
        
        print(f"Multimodal analysis result:")
        print(f"  Number of analyzed frames: {len(multimodal_results.get('frames', []))}")
        
        # Check analysis results
        if 'frames' in multimodal_results:
            for frame_result in multimodal_results['frames'][:2]:  # Only display first 2
                print(f"     Frame: {frame_result.get('frame', 'unknown')}")
                if 'yolo' in frame_result:
                    print(f"      Number of YOLO objects: {len(frame_result['yolo'].get('objects', []))}")
                if 'blip' in frame_result:
                    print(f"      BLIP description: {frame_result['blip'].get('description', '')[:50]}...")
    
    def test_csv_output_generation(self):
        """Test CSV output generation"""
        print("\n=== Test CSV Output Generation ===")
        
        # Create test data
        test_data = {
            'video_name': ['test_video_1', 'test_video_2'],
            'video_path': ['path1.mp4', 'path2.mp4'],
            'duration': [10.5, 15.2],
            'width': [1920, 1280],
            'height': [1080, 720],
            'frame_rate': [30.0, 25.0],
            'file_size': [1024000, 2048000],
            'has_speech': [True, False],
            'speech_text': ['Test speech 1', ''],
            'speech_text_length': [5, 0],
            'keyframe_count': [5, 3],
            'representative_frame_count': [3, 2]
        }
        
        test_df = pd.DataFrame(test_data)
        
        # Save CSV
        csv_path = os.path.join(self.test_output_dir, "test_csv_output.csv")
        test_df.to_csv(csv_path, index=False)
        
        # Validate file
        self.assertTrue(os.path.exists(csv_path))
        file_size = os.path.getsize(csv_path)
        print(f"CSV file generation successful:")
        print(f"   File path: {csv_path}")
        print(f"   File size: {file_size} bytes")
        self.assertGreater(file_size, 0)
        
        # Read and validate content
        read_df = pd.read_csv(csv_path)
        self.assertEqual(len(read_df), len(test_df))
        self.assertEqual(len(read_df.columns), len(test_df.columns))
        
        print(f"   Row count: {len(read_df)}")
        print(f"   Column count: {len(read_df.columns)}")
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n=== Test Error Handling ===")
        
        # Test nonexistent video file
        nonexistent_path = "data/tiktok_videos/nonexistent.mp4"
        
        # Should be able to handle error without crashing
        try:
            metadata = self.extractor.extract_video_metadata(nonexistent_path)
            print("Error handling test passed")
        except Exception as e:
            print(f"Error handling test failed: {e}")
            self.fail("Should be able to handle nonexistent file")

def run_tests():
    """Run all tests"""
    print("Starting feature extractor functionality tests...")
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestFeatureExtractor)
    
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