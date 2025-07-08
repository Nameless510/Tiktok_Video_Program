import os
import ffmpeg
import hashlib
import re
import subprocess
import json

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

    def extract_speech_highlight_frames(self, video_path, output_dir, speech_segments, confidence_threshold=0.7):
        highlight_times = []
        for seg in speech_segments:
            if seg.get('confidence', 0) >= confidence_threshold:
                t = (seg['start'] + seg['end']) / 2
                highlight_times.append(t)
        highlight_files = []
        for idx, t in enumerate(highlight_times):
            out_path = os.path.join(output_dir, f"speech_highlight_{idx:03d}_{t:.2f}.jpg")
            try:
                (
                    ffmpeg
                    .input(video_path, ss=t)
                    .output(out_path, vframes=1)
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
                highlight_files.append({'file': out_path, 'timestamp': t})
            except Exception as e:
                print(f"Error extracting speech highlight frame at {t}s: {e}")
        return highlight_files

    def extract_keyframes(self, video_path, output_dir, speech_segments=None):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        keyframes_dir = output_dir
        os.makedirs(keyframes_dir, exist_ok=True)

        # Get I-frame timestamps using ffprobe
        def get_i_frame_timestamps(video_path):
            cmd = [
                'ffprobe', '-select_streams', 'v',
                '-show_frames', '-show_entries', 'frame=pict_type,pkt_pts_time',
                '-of', 'json', video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            frames_json = json.loads(result.stdout)
            i_frame_times = []
            for frame in frames_json.get('frames', []):
                if frame.get('pict_type') == 'I':
                    try:
                        i_frame_times.append(float(frame['pkt_pts_time']))
                    except Exception:
                        continue
            return i_frame_times

        i_frame_times = get_i_frame_timestamps(video_path)
        for idx, t in enumerate(i_frame_times):
            out_path = os.path.join(keyframes_dir, f"{video_name}_keyframe_{idx+1:04d}_{t:.2f}.jpg")
            try:
                (
                    ffmpeg
                    .input(video_path, ss=t)
                    .output(out_path, vframes=1)
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
            except Exception as e:
                print(f"Error extracting I-frame at {t}s: {e}")

        # Sampled frames (1 fps)
        duration = 0
        try:
            probe = ffmpeg.probe(video_path)
            duration = float(probe['format']['duration'])
        except Exception as e:
            print(f"Error getting video duration: {e}")
        fps = 1
        sampled_times = [round(i, 2) for i in range(0, int(duration), fps)]
        for idx, t in enumerate(sampled_times):
            out_path = os.path.join(keyframes_dir, f"{video_name}_sampled_{idx+1:04d}_{t:.2f}.jpg")
            try:
                (
                    ffmpeg
                    .input(video_path, ss=t)
                    .output(out_path, vframes=1)
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
            except Exception as e:
                print(f"Error extracting sampled frame at {t}s: {e}")

        # Highlight frames
        if speech_segments is not None:
            highlight_frames = self.extract_speech_highlight_frames(video_path, output_dir, speech_segments, confidence_threshold=0.7)
            for idx, item in enumerate(highlight_frames):
                t = item['timestamp']
                out_path = os.path.join(output_dir, f"speech_highlight_{idx:03d}_{t:.2f}.jpg")
                os.rename(item['file'], out_path)
                item['file'] = out_path

        keyframe_count = len([f for f in os.listdir(keyframes_dir) if f.endswith('.jpg')])
        return keyframes_dir, keyframe_count 