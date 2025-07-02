import os
import ffmpeg
import whisper
import torch
import torchaudio
import numpy as np
from pathlib import Path

class AudioProcessor:
    """Audio processing: speech detection, extraction, and transcription."""
    def __init__(self):
        self._whisper_model = None
        self._vad_model = None

    def get_vad_model(self):
        if self._vad_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading VAD model on device: {device}")
            try:
                self._vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                     model='silero_vad',
                                                     force_reload=False,
                                                     onnx=False)
                self._vad_model = self._vad_model.to(device)
                self._vad_model.eval()
            except Exception as e:
                print(f"Error loading VAD model: {e}")
                return None
        return self._vad_model

    def get_whisper_model(self):
        if self._whisper_model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading Whisper model on device: {device}")
            try:
                self._whisper_model = whisper.load_model("base", device=device)
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
                return None
        return self._whisper_model

    def extract_audio(self, video_path, output_dir):
        """Extract audio from video with improved error handling and validation."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(output_dir, f'{video_name}.wav')
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Check if video file exists and is readable
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return None
                
            # Get video info to check if it has audio
            probe = ffmpeg.probe(video_path)
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            
            if not audio_streams:
                print(f"No audio stream found in video: {video_path}")
                return None
            
            print(f"Extracting audio from {video_path} to {audio_path}")
            
            # Extract audio with better parameters
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, 
                       acodec='pcm_s16le',  # 16-bit PCM
                       ac=1,                # mono channel
                       ar='16000',          # 16kHz sample rate
                       vn=None,             # no video
                       loglevel='error')    # reduce log output
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            
            # Validate extracted audio file
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                print(f"Audio extracted successfully: {audio_path}")
                return audio_path
            else:
                print(f"Audio extraction failed: file is empty or not created")
                return None
                
        except ffmpeg.Error as e:
            print(f"FFmpeg error extracting audio from {video_path}: {e.stderr.decode() if e.stderr else str(e)}")
            return None
        except Exception as e:
            print(f"Error extracting audio from {video_path}: {e}")
            return None

    def has_speech(self, audio_path, threshold=0.5):
        """Check if audio contains speech with improved validation."""
        if not audio_path or not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return False
            
        if os.path.getsize(audio_path) == 0:
            print(f"Audio file is empty: {audio_path}")
            return False
            
        try:
            vad_model = self.get_vad_model()
            if vad_model is None:
                print("VAD model not available, skipping speech detection")
                return False
                
            device = next(vad_model.parameters()).device
            print(f"Loading audio for speech detection: {audio_path}")
            
            # Load audio with error handling
            waveform, sample_rate = torchaudio.load(audio_path)
            print(f"Audio loaded: shape={waveform.shape}, sample_rate={sample_rate}")
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                print("Converted stereo to mono")
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
                print(f"Resampled to {sample_rate}Hz")
            
            waveform = waveform.to(device)
            chunk_size = 512
            speech_probs = []
            
            total_samples = waveform.shape[-1]
            if total_samples < chunk_size:
                padding = chunk_size - total_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                total_samples = chunk_size
            
            # Process audio in chunks
            for i in range(0, total_samples, chunk_size):
                chunk = waveform[:, i:i+chunk_size]
                if chunk.shape[-1] == chunk_size:
                    with torch.no_grad():
                        chunk_probs = vad_model(chunk, sample_rate).cpu().detach().numpy()
                        speech_probs.extend(chunk_probs.flatten())
            
            if not speech_probs:
                print("No speech probabilities calculated")
                return False
                
            speech_frames = (torch.tensor(speech_probs) > threshold).sum().item()
            speech_percentage = speech_frames / len(speech_probs)
            
            print(f"Speech detection: {speech_percentage:.2%} frames contain speech")
            return speech_percentage > 0.1
            
        except Exception as e:
            print(f"Error in speech detection for {audio_path}: {e}")
            return False

    def speech_to_text(self, audio_path):
        """Transcribe speech to text with improved error handling."""
        if not audio_path or not os.path.exists(audio_path):
            print(f"Audio file not found for transcription: {audio_path}")
            return ""
            
        if os.path.getsize(audio_path) == 0:
            print(f"Audio file is empty for transcription: {audio_path}")
            return ""
            
        try:
            model = self.get_whisper_model()
            if model is None:
                print("Whisper model not available, skipping transcription")
                return ""
                
            print(f"Transcribing audio: {audio_path}")
            result = model.transcribe(audio_path)
            transcribed_text = result.get("text", "").strip()
            
            if transcribed_text:
                print(f"Transcription successful: {len(transcribed_text)} characters")
            else:
                print("No text transcribed")
                
            return transcribed_text
            
        except Exception as e:
            print(f"Error transcribing audio {audio_path}: {e}")
            return ""

    def process_audio_for_video(self, video_path, output_dir):
        """Complete audio processing pipeline for a video."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Extract audio
        audio_path = self.extract_audio(video_path, output_dir)
        if not audio_path:
            return {
                'audio_path': '',
                'has_speech': False,
                'speech_text': '',
                'speech_text_length': 0
            }
        
        # Check for speech
        has_speech = self.has_speech(audio_path)
        
        # Transcribe if speech detected
        speech_text = ""
        if has_speech:
            speech_text = self.speech_to_text(audio_path)
            if speech_text:
                self.save_speech_text(output_dir, speech_text)
        
        return {
            'audio_path': audio_path,
            'has_speech': has_speech,
            'speech_text': speech_text,
            'speech_text_length': len(speech_text)
        }

    def save_speech_text(self, video_output_dir, speech_text):
        """Save transcribed speech text to file."""
        txt_file_path = os.path.join(video_output_dir, 'speech_text.txt')
        try:
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(speech_text)
            print(f"Speech text saved to: {txt_file_path}")
        except Exception as e:
            print(f"Error saving speech text to {txt_file_path}: {e}")

    def read_speech_text(self, txt_path):
        """Read transcribed speech text from file."""
        if not txt_path or not os.path.exists(txt_path):
            return ""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading speech text from {txt_path}: {e}")
            return "" 