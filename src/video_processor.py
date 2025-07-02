import os
import ffmpeg

class VideoProcessor:
    """Video processing: metadata extraction and keyframe extraction."""
    def get_video_metadata(self, video_path):
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            width = int(video_info['width'])
            height = int(video_info['height'])
            if 'r_frame_rate' in video_info:
                frame_rate_str = video_info['r_frame_rate']
                if '/' in frame_rate_str:
                    num, den = map(int, frame_rate_str.split('/'))
                    frame_rate = num / den if den != 0 else 0
                else:
                    frame_rate = float(frame_rate_str)
            else:
                frame_rate = 0
            file_size = int(probe['format']['size'])
            return {
                'duration': duration,
                'width': width,
                'height': height,
                'frame_rate': frame_rate,
                'file_size': file_size
            }
        except Exception as e:
            print(f"Error extracting metadata from {video_path}: {e}")
            return {
                'duration': 0,
                'width': 0,
                'height': 0,
                'frame_rate': 0,
                'file_size': 0
            }

    def extract_keyframes(self, video_path, output_dir):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        keyframes_dir = output_dir
        os.makedirs(keyframes_dir, exist_ok=True)
        output_pattern = os.path.join(keyframes_dir, f'{video_name}_keyframe_%04d.jpg')
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_pattern, **{
                    'vf': "select='eq(pict_type,PICT_TYPE_I)'",
                    'vsync': 'vfr',
                    'frame_pts': 'true'
                })
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            saved_files = [f for f in os.listdir(keyframes_dir) if f.endswith('.jpg') and f.startswith(f'{video_name}_keyframe_')]
            return keyframes_dir, len(saved_files)
        except Exception as e:
            print(f"Error extracting keyframes from {video_path}: {e}")
            return keyframes_dir, 0 