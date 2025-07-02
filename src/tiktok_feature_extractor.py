import os
import json
import pandas as pd
import time
from tqdm import tqdm
from audio_processor import AudioProcessor
from video_processor import VideoProcessor
from frame_analyzer import FrameAnalyzer
from multimodal_extractor import MultimodalExtractor

class TikTokFeatureExtractor:
    """Main controller for TikTok video feature extraction."""
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.frame_analyzer = FrameAnalyzer()
        self.multimodal_extractor = MultimodalExtractor()

    def extract_video_features(self, video_path, output_dir):
        """Extract comprehensive features from a single video with timing."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nProcessing video: {video_name}")
        
        # Initialize timing dictionary
        timing_stats = {}
        total_start_time = time.time()
        
        features = {
            'video_path': video_path,
            'duration': 0,
            'width': 0,
            'height': 0,
            'frame_rate': 0,
            'file_size': 0,
            'audio_path': '',
            'has_speech': False,
            'speech_text_length': 0,
            'speech_text': '',
            'keyframe_count': 0,
            'representative_frame_count': 0,
            'text_frame_count': 0,
            'multimodal_features_path': '',
            'yolo_summary': '',
            'blip_summary': ''
        }
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Extract video metadata
            print("  - Extracting video metadata...")
            start_time = time.time()
            metadata = self.video_processor.get_video_metadata(video_path)
            features.update(metadata)
            timing_stats['video_metadata'] = time.time() - start_time
            print(f"    ✓ Video metadata: {timing_stats['video_metadata']:.2f}s")
            
            # 2. Process audio (extract, detect speech, transcribe)
            print("  - Processing audio...")
            start_time = time.time()
            audio_results = self.audio_processor.process_audio_for_video(video_path, output_dir)
            features.update(audio_results)
            timing_stats['audio_processing'] = time.time() - start_time
            print(f"    ✓ Audio processing: {timing_stats['audio_processing']:.2f}s")
            
            # 3. Extract keyframes
            print("  - Extracting keyframes...")
            start_time = time.time()
            keyframes_dir, keyframe_count = self.video_processor.extract_keyframes(video_path, output_dir)
            features['keyframe_count'] = keyframe_count
            timing_stats['keyframe_extraction'] = time.time() - start_time
            print(f"    ✓ Keyframe extraction: {timing_stats['keyframe_extraction']:.2f}s")
            
            if keyframe_count > 0:
                # 4. Filter similar keyframes
                print("  - Filtering similar keyframes...")
                start_time = time.time()
                filtered_frames = self.frame_analyzer.filter_similar_keyframes(keyframes_dir, video_name)
                timing_stats['frame_filtering'] = time.time() - start_time
                print(f"    ✓ Frame filtering: {timing_stats['frame_filtering']:.2f}s")
                print(f"    - Kept {len(filtered_frames)} frames after filtering")
                
                # 5. Select representative frames
                print("  - Selecting representative frames...")
                start_time = time.time()
                representative_frames = self.frame_analyzer.get_representative_frames(keyframes_dir, video_name)
                timing_stats['representative_selection'] = time.time() - start_time
                features['representative_frame_count'] = len(representative_frames)
                print(f"    ✓ Representative selection: {timing_stats['representative_selection']:.2f}s")
                
                # 6. Save representative frames
                print("  - Saving representative frames...")
                start_time = time.time()
                self.frame_analyzer.save_representative_frames(representative_frames, output_dir, video_name)
                timing_stats['frame_saving'] = time.time() - start_time
                print(f"    ✓ Frame saving: {timing_stats['frame_saving']:.2f}s")
                
                # 7. Extract multimodal features
                print("  - Extracting multimodal features...")
                start_time = time.time()
                multimodal_features = self.multimodal_extractor.extract_multimodal_features_for_frames(keyframes_dir, video_name)
                timing_stats['multimodal_extraction'] = time.time() - start_time
                print(f"    ✓ Multimodal extraction: {timing_stats['multimodal_extraction']:.2f}s")
                
                # Save multimodal features
                start_time = time.time()
                multimodal_path = os.path.join(output_dir, "multimodal_features.json")
                with open(multimodal_path, "w", encoding="utf-8") as f:
                    json.dump(multimodal_features, f, ensure_ascii=False, indent=2)
                features['multimodal_features_path'] = multimodal_path
                timing_stats['multimodal_saving'] = time.time() - start_time
                print(f"    ✓ Multimodal saving: {timing_stats['multimodal_saving']:.2f}s")
                
                # Generate summaries
                start_time = time.time()
                yolo_summary, blip_summary = self.multimodal_extractor.summarize_multimodal_features(multimodal_features)
                features['yolo_summary'] = yolo_summary
                features['blip_summary'] = blip_summary
                timing_stats['summary_generation'] = time.time() - start_time
                print(f"    ✓ Summary generation: {timing_stats['summary_generation']:.2f}s")
                
                # Count text frames (removed OCR dependency)
                features['text_frame_count'] = 0
            else:
                timing_stats['frame_filtering'] = 0
                timing_stats['representative_selection'] = 0
                timing_stats['frame_saving'] = 0
                timing_stats['multimodal_extraction'] = 0
                timing_stats['multimodal_saving'] = 0
                timing_stats['summary_generation'] = 0
            
            # Calculate total time
            total_time = time.time() - total_start_time
            timing_stats['total_time'] = total_time
            
            # Print simple timing summary
            print(f"\n⏱️  Time Summary for {video_name}:")
            print(f"  Total time: {total_time:.2f}s")
            # Show the slowest step
            if timing_stats:
                slowest_step = max([(k, v) for k, v in timing_stats.items() if k != 'total_time'], key=lambda x: x[1])
                print(f"  Slowest step: {slowest_step[0].replace('_', ' ')} ({slowest_step[1]:.2f}s)")
            
            # Add timing info to features
            features['processing_time'] = total_time
            features['timing_stats'] = timing_stats
            
            print(f"  ✓ Completed processing {video_name}")
            
        except Exception as e:
            print(f"  ✗ Error processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            
        return features

    def extract_features(self, input_path, output_folder, csv_output_path=None):
        """Extract features from multiple videos or a single video."""
        # Determine input type and get video files
        if os.path.isfile(input_path) and input_path.lower().endswith('.mp4'):
            video_files = [input_path]
        else:
            video_files = []
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith('.mp4'):
                        video_files.append(os.path.join(root, file))
        
        print(f"Found {len(video_files)} video files to process")
        
        if not video_files:
            print("No video files found!")
            return pd.DataFrame()
        
        # Process each video
        all_features = []
        for video_path in tqdm(video_files, desc='Processing videos'):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_output_dir = os.path.join(output_folder, video_name)
            
            features = self.extract_video_features(video_path, video_output_dir)
            all_features.append(features)
        
        # Create DataFrame with proper column order
        df = pd.DataFrame(all_features)
        
        # Ensure all expected columns exist
        expected_columns = [
            'video_path', 'duration', 'width', 'height', 'frame_rate', 'file_size',
            'audio_path', 'has_speech', 'speech_text_length', 'speech_text',
            'keyframe_count', 'representative_frame_count', 'text_frame_count',
            'multimodal_features_path', 'yolo_summary', 'blip_summary'
        ]
        
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns
        df = df[expected_columns]
        
        # Save results
        if csv_output_path:
            df.to_csv(csv_output_path, index=False, encoding='utf-8')
            print(f"\nResults saved to: {csv_output_path}")
        
        return df

    def extract_features_from_folder(self, video_folder, output_folder, csv_output_path=None):
        return self.extract_features(video_folder, output_folder, csv_output_path)

    def extract_features_from_single_video(self, video_path, output_folder, csv_output_path=None):
        return self.extract_features(video_path, output_folder, csv_output_path)