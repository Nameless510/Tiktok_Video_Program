import os
import ffmpeg
import whisper
import torch
import torchaudio
import numpy as np
from pathlib import Path
import subprocess

class AudioProcessor:
    """Audio processing: speech detection, extraction, and transcription."""
    def __init__(self):
        self._whisper_model = None
        self._vad_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_vad_model(self):
        if self._vad_model is None:
            print(f"Loading VAD model on device: {self.device}")
            try:
                # torch.hub.load returns a tuple (model, utils, ...)
                vad_package = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                     model='silero_vad',
                                                     force_reload=False,
                                                     onnx=False)
                # Extract the model from the tuple
                self._vad_model = vad_package[0] if isinstance(vad_package, tuple) else vad_package
                self._vad_model = self._vad_model.to(self.device)
                self._vad_model.eval()
            except Exception as e:
                print(f"Error loading VAD model: {e}")
                return None
        return self._vad_model

    def get_whisper_model(self):
        if self._whisper_model is None:
            print(f"Loading Whisper model on device: {self.device}")
            try:
                self._whisper_model = whisper.load_model("base", device=self.device)
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
                return None
        return self._whisper_model

    def extract_audio(self, video_path, output_path):
        """Extract audio from video file."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Extract audio using ffmpeg
            command = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                return output_path
            else:
                print(f"FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return None

    def has_speech(self, audio_path):
        """Detect if audio contains speech using VAD."""
        try:
            # Load VAD model
            vad_model = self.get_vad_model()
            if vad_model is None:
                print("VAD model not available, skipping speech detection")
                return False
                
            # Load audio
            wav, sr = torchaudio.load(audio_path)
            wav = wav.to(self.device)
            wav = wav.to(torch.float32) 
            
            # Resample if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
                sr = 16000
            
            # Silero VAD expects chunks of 512 samples for 16kHz
            chunk_size = 512
            speech_detected = False
            
            # Process audio in chunks
            for i in range(0, wav.shape[-1], chunk_size):
                chunk = wav[..., i:i+chunk_size]
                
                # Pad the last chunk if it's smaller than chunk_size
                if chunk.shape[-1] < chunk_size:
                    padding = chunk_size - chunk.shape[-1]
                    chunk = torch.nn.functional.pad(chunk, (0, padding))
                
                # Detect speech in this chunk
                with torch.no_grad():
                    speech_prob = vad_model(chunk, sr).item()
                
                if speech_prob > 0.5:
                    speech_detected = True
                    break  # Found speech, no need to continue
            
            return speech_detected
            
        except Exception as e:
            print(f"Speech detection error: {e}")
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
            wav, sr = torchaudio.load(audio_path)
            wav = wav.to(torch.float32)  
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
        """Process audio for a video: extract, detect speech, transcribe."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        results = {
                'has_speech': False,
                'speech_text': '',
                'speech_text_length': 0
            }
        
        try:
            # Extract audio
            audio_path = os.path.join(output_dir, f"{video_name}.wav")
            self.extract_audio(video_path, audio_path)
            
            # Detect speech
            has_speech = self.has_speech(audio_path)
            results['has_speech'] = has_speech
        
            # Transcribe if speech detected
            if has_speech:
                transcript = self.speech_to_text(audio_path)
                results['speech_text'] = transcript
                results['speech_text_length'] = len(transcript)
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            
        return results

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