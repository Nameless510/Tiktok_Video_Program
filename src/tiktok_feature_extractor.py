import os
import pandas as pd
import time
from tqdm import tqdm
from audio_processor import AudioProcessor
from video_processor import VideoProcessor
from frame_analyzer import FrameAnalyzer
from multimodal_extractor import MultimodalExtractor

class TikTokFeatureExtractor:
    """Main controller for TikTok video feature extraction with Qwen-VL integration."""
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.video_processor = VideoProcessor()
        self.frame_analyzer = FrameAnalyzer()
        self.multimodal_extractor = MultimodalExtractor()

    def extract_video_features(self, video_path, output_dir):
        """Extract comprehensive features from a single video with Qwen-VL analysis."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        features = {
            'video_name': video_name,
            'video_path': video_path,
            'duration': 0,
            'width': 0,
            'height': 0,
            'frame_rate': 0,
            'file_size': 0,
            'has_speech': False,
            'speech_text': '',
            'keyframe_count': 0,
            'representative_frame_count': 0,
            'video_description': '',
            'primary_category': '',
            'secondary_category': '',
            'tertiary_category': '',
            'processing_time': 0
        }
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            total_start_time = time.time()
            
            # 1. Extract video metadata
            metadata = self.video_processor.get_video_metadata(video_path)
            features.update(metadata)
            
            # 2. Process audio
            audio_results = self.audio_processor.process_audio_for_video(video_path, output_dir)
            features.update(audio_results)

            # 2.5. Extract speech_segments for smart frame extraction
            speech_segments = None
            if isinstance(audio_results, dict):
                speech_segments = (
                    audio_results.get('structured_data', {})
                    .get('speech_analysis', {})
                    .get('segments', [])
                )

            # 3. Extract keyframes (with speech_segments for highlight frames)
            keyframes_dir, keyframe_count = self.video_processor.extract_keyframes(video_path, output_dir, speech_segments=speech_segments)
            features['keyframe_count'] = keyframe_count

            if keyframe_count > 0:
                # 4. Filter similar keyframes
                filtered_frames = self.frame_analyzer.filter_similar_keyframes(keyframes_dir, video_name)
                
                # 5. Select representative frames
                representative_frames = self.frame_analyzer.get_representative_frames(keyframes_dir, video_name)
                features['representative_frame_count'] = len(representative_frames)
                
                # 6. Save representative frames
                self.frame_analyzer.save_representative_frames(representative_frames, output_dir, video_name)
                self.frame_analyzer.save_representative_timestamps(output_dir, video_name)

                # 8. Extract Qwen-VL features
                qwen_results = self.multimodal_extractor.extract_qwen_features(output_dir, video_name, audio_transcript=audio_results.get('speech_text', ''))
                
                # Update features with Qwen results
                if qwen_results:
                    features.update(qwen_results)
            
            # Calculate total time
            total_time = time.time() - total_start_time
            features['processing_time'] = total_time
            
        except Exception as e:
            print(f"  ✗ Error processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            
        return features

    def extract_features_from_folder(self, video_folder, output_folder, csv_output_path=None):
        """Extract features from all videos in a folder."""
        video_files = []
        for file in os.listdir(video_folder):
            if file.lower().endswith('.mp4'):
                video_files.append(file)
        
        print(f"Found {len(video_files)} video files to process")
        
        if not video_files:
            print("No video files found!")
            return pd.DataFrame()
        
        # Process each video
        all_features = []
        for video_file in tqdm(video_files, desc='Processing videos'):
            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_dir = os.path.join(output_folder, video_name)
            
            features = self.extract_video_features(video_path, video_output_dir)
            all_features.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Save to CSV if path provided
        if csv_output_path and not df.empty:
            df.to_csv(csv_output_path, index=False, encoding='utf-8')
            print(f"\n💾 Results saved to: {csv_output_path}")
        
        return df

    def extract_features_from_single_video(self, video_path, output_folder, csv_output_path=None):
        """Extract features from a single video."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_folder, video_name)
        
        features = self.extract_video_features(video_path, video_output_dir)
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Save to CSV if path provided
        if csv_output_path and not df.empty:
            df.to_csv(csv_output_path, index=False, encoding='utf-8')
            print(f"\n💾 Results saved to: {csv_output_path}")
        
        return df