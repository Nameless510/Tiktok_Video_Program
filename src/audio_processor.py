import os
import ffmpeg
import whisper
import torch
import torchaudio
import numpy as np
from pathlib import Path
import subprocess
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Any
import re
import json
from dataclasses import dataclass
from scipy import signal
from scipy.spatial.distance import cosine
import hashlib

from shared_models import whisper_model, silero_vad_model, demucs_model, shazam_client

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    chunk_duration: float = 30.0
    overlap_duration: float = 2.0
    vad_threshold: float = 0.2
    noise_reduce_strength: float = 0.1
    whisper_model_size: str = "large-v3"
    language: str = "en"
    task: str = "transcribe"

class AudioSeparator:
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def separate_audio_sources(self, audio_path: str, output_dir: str) -> Dict[str, str]:
        try:
            model = demucs_model
            if model is None:
                return {"vocals": audio_path, "non_vocals": audio_path}
            from demucs import apply
            wav, sr = torchaudio.load(audio_path)
            wav = wav.to(self.device)
            sr = int(sr)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  
            if wav.size(0) == 1: 
                wav = wav.repeat(2, 1)
            if wav.dim() == 2:
                wav = wav.unsqueeze(0)  
            with torch.no_grad():
                sources = apply.apply_model(model, wav, device=self.device)[0]  # (sources, channels, samples)
            sources_names = getattr(model, 'sources', ["drums", "bass", "other", "vocals"])
            vocals_idx = None
            for i, name in enumerate(sources_names):
                if name.lower() == "vocals":
                    vocals_idx = i
                    break
            if vocals_idx is None:
                raise ValueError("Demucs model does not provide vocals stem.")
            non_vocals_indices = [i for i in range(len(sources_names)) if i != vocals_idx]
            vocals_audio = sources[vocals_idx]
            non_vocals_audio = sum([sources[i] for i in non_vocals_indices])
            vocals_path = os.path.join(output_dir, "vocals.wav")
            non_vocals_path = os.path.join(output_dir, "non_vocals.wav")
            torchaudio.save(vocals_path, vocals_audio.cpu(), sr)
            torchaudio.save(non_vocals_path, non_vocals_audio.cpu(), sr)
            print(f"Two-stems separated and saved to: {vocals_path}, {non_vocals_path}")
            return {"vocals": vocals_path, "non_vocals": non_vocals_path}
        except Exception as e:
            print(f"Audio separation error: {e}")
            return {"vocals": audio_path, "non_vocals": audio_path}

class SpeechAnalyzer:
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def analyze_speech_content(self, audio_path: str) -> Dict[str, Any]:
        try:
            if whisper_model is None:
                return {"text": "", "processed_text": ""}
            transcription_result = self._transcribe_speech(audio_path)
            processed_text = self._post_process_text(transcription_result.get("text", ""))
            return {
                "text": transcription_result.get("text", ""),
                "processed_text": processed_text
            }
        except Exception as e:
            print(f"Speech analysis error: {e}")
            return {"text": "", "processed_text": ""}
    
    def _transcribe_speech(self, audio_path: str) -> Dict:
        try:
            if whisper_model is None:
                return {"text": "", "segments": [], "language": "en"}
            
            options = {
                "language": self.config.language,
                "task": self.config.task,
                "fp16": False,
                "condition_on_previous_text": True,
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            
            result = whisper_model.transcribe(audio_path, **options)
            return result
            
        except Exception as e:
            print(f"Speech transcription error: {e}")
            return {"text": "", "segments": [], "language": "en"}
    
    def _post_process_text(self, text: str) -> str:
        try:
            filler_patterns = [
                r'\b(um|uh|er|ah|hmm|mm|mhm|uh-huh|uh-uh)\b',
                r'\b(like|you know|i mean|basically|actually|literally)\b',
                r'\b(so|well|right|okay|ok)\b',
                r'\s+',
            ]
            
            for pattern in filler_patterns:
                text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
            
            error_corrections = {
                r'\b(its|it\'s)\b': "it's",
                r'\b(they\'re|their|there)\b': "they're",
                r'\b(you\'re|your)\b': "you're",
                r'\b(we\'re|were)\b': "we're",
                r'\b(can\'t|cant)\b': "can't",
                r'\b(won\'t|wont)\b': "won't",
                r'\b(don\'t|dont)\b': "don't",
            }
            
            for pattern, replacement in error_corrections.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            text = ' '.join(text.split())
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
            
            return text
            
        except Exception as e:
            print(f"Text post-processing error: {e}")
            return text

class MusicAnalyzer:
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def analyze_music_copyright(self, audio_path: str) -> Dict[str, Any]:
        try:
            if shazam_client is not None:
                import asyncio
                async def recognize_audio():
                    return await shazam_client.recognize(audio_path)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        import nest_asyncio
                        nest_asyncio.apply()
                        task = asyncio.ensure_future(recognize_audio())
                        shazam_result = loop.run_until_complete(task)
                    else:
                        shazam_result = loop.run_until_complete(recognize_audio())
                except Exception as e:
                    print(f"Asyncio fallback: {e}")
                    shazam_result = {}
            else:
                shazam_result = {}
            return {"audio_fingerprint": "", "shazam_result": shazam_result}
        except Exception as e:
            print(f"Music analysis error: {e}")
            return {"audio_fingerprint": "", "shazam_result": {}}

class EventDetector:
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def detect_audio_events(self, audio_path: str) -> List[Dict]:
        try:
            y, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            sr = int(sr)
            events = []
            sound_events = self._detect_sound_events(y, sr)
            events.extend(sound_events)
            music_events = self._detect_music_events(y, sr)
            events.extend(music_events)
            speech_events = self._detect_speech_events(y, sr)
            events.extend(speech_events)
            ambient_events = self._detect_ambient_events(y, sr)
            events.extend(ambient_events)
            events.sort(key=lambda x: x["start_time"])
            return events
        except Exception as e:
            print(f"Event detection error: {e}")
            return []
    
    def _detect_sound_events(self, y: np.ndarray, sr: int) -> List[Dict]:
        sr = int(sr)
        try:
            events = []
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            for i, time in enumerate(onset_times):
                start_frame = onset_frames[i]
                end_frame = start_frame + int(0.5 * sr / 512)
                if end_frame < len(y) // 512:
                    segment = y[start_frame * 512:end_frame * 512]
                    energy = float(np.mean(librosa.feature.rms(y=segment)))
                    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
                    event_type = self._classify_sound_event(float(energy), float(spectral_centroid))
                    confidence = float(min(0.9, float(energy) * 10))
                    if confidence > 0.2:
                        events.append({
                            "type": "sound_effect",
                            "subtype": event_type,
                            "start_time": float(time),
                            "end_time": float(time + 0.5),
                            "confidence": confidence,
                            "features": {
                                "energy": float(energy),
                                "spectral_centroid": float(spectral_centroid)
                            }
                        })
            return events
        except Exception as e:
            print(f"Sound event detection error: {e}")
            return []
    
    def _detect_music_events(self, y: np.ndarray, sr: int) -> List[Dict]:
        sr = int(sr)
        try:
            events = []
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            rec_mat = librosa.segment.recurrence_matrix(chroma)
            from scipy.signal import find_peaks
            diag = np.mean(rec_mat, axis=0)
            peaks, _ = find_peaks(diag, distance=sr//512)
            segment_times = librosa.frames_to_time(peaks, sr=sr)
            for i, time in enumerate(beat_times[::4]):
                events.append({
                    "type": "music",
                    "subtype": "beat",
                    "start_time": float(time),
                    "end_time": float(time + 0.1),
                    "confidence": 0.8,
                    "features": {"tempo": float(tempo)}
                })
            for seg_time in segment_times:
                events.append({
                    "type": "music",
                    "subtype": "segment",
                    "start_time": float(seg_time),
                    "end_time": float(seg_time + 0.5),
                    "confidence": 0.7,
                    "features": {"method": "recurrence_peaks"}
                })
            return events
        except Exception as e:
            print(f"Music event detection error: {e}")
            return []
    
    def _detect_speech_events(self, y: np.ndarray, sr: int) -> List[Dict]:
        sr = int(sr)
        try:
            events = []
            rms = librosa.feature.rms(y=y)
            rms_times = librosa.frames_to_time(range(len(rms[0])), sr=sr)
            speech_threshold = float(np.mean(rms)) * 1.5
            speech_frames = rms[0] > speech_threshold
            speech_segments = []
            start_frame = None
            for i, is_speech in enumerate(speech_frames):
                if is_speech and start_frame is None:
                    start_frame = i
                elif not is_speech and start_frame is not None:
                    speech_segments.append((start_frame, i))
                    start_frame = None
            if start_frame is not None:
                speech_segments.append((start_frame, len(speech_frames)))
            for start_frame, end_frame in speech_segments:
                start_time = rms_times[start_frame]
                end_time = rms_times[min(end_frame, len(rms_times) - 1)]
                if end_time - start_time > 0.5:
                    events.append({
                        "type": "speech",
                        "subtype": "voice",
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "confidence": 0.7,
                        "features": {"duration": float(end_time - start_time)}
                    })
            return events
        except Exception as e:
            print(f"Speech event detection error: {e}")
            return []
    
    def _detect_ambient_events(self, y: np.ndarray, sr: int) -> List[Dict]:
        sr = int(sr)
        try:
            events = []
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_diff = np.diff(mfcc, axis=1)
            change_threshold = float(np.std(mfcc_diff)) * 2
            change_frames = np.where(np.mean(np.abs(mfcc_diff), axis=0) > change_threshold)[0]
            for frame in change_frames[::int(sr/2)]:
                time = librosa.frames_to_time(frame, sr=sr)
                events.append({
                    "type": "ambient",
                    "subtype": "environment_change",
                    "start_time": float(time),
                    "end_time": float(time + 0.2),
                    "confidence": 0.6,
                    "features": {"change_magnitude": float(np.mean(np.abs(mfcc_diff[:, frame])))}
                })
            return events
        except Exception as e:
            print(f"Ambient event detection error: {e}")
            return []
    
    def _classify_sound_event(self, energy: float, spectral_centroid: float) -> str:
        energy = float(energy)
        spectral_centroid = float(spectral_centroid)
        if energy > 0.1:
            if spectral_centroid > 3000:
                return "high_frequency_impact"
            elif spectral_centroid > 1500:
                return "medium_frequency_impact"
            else:
                return "low_frequency_impact"
        else:
            return "subtle_sound"

class AudioAggregator:
    
    def __init__(self, config: AudioConfig):
        self.config = config
    
    def aggregate_results(self, 
                         preprocessed_audio: str,
                         separated_sources: Dict[str, str],
                         speech_analysis: Dict[str, Any],
                         music_analysis: Dict[str, Any],
                         audio_events: List[Dict]) -> Dict[str, Any]:
        try:
            aligned_results = self._align_timeline(
                speech_analysis, music_analysis, audio_events
            )
            
            structured_data = self._create_structured_data(
                preprocessed_audio,
                separated_sources,
                aligned_results
            )
            
            summary = self._generate_summary(structured_data)
            
            return {
                "structured_data": structured_data,
                "summary": summary,
                "metadata": {
                    "processing_time": 0,  
                    "audio_duration": self._get_audio_duration(preprocessed_audio),
                    "analysis_modules": ["preprocessor", "separator", "speech", "music", "events", "aggregator"],
                }
            }
            
        except Exception as e:
            print(f"Result aggregation error: {e}")
            return {"structured_data": {}, "summary": {}, "metadata": {}}
    
    def _align_timeline(self, speech_analysis: Dict, music_analysis: Dict, audio_events: List[Dict]) -> Dict:
        try:
            aligned = {
                "timeline": [],
                "speech_segments": speech_analysis.get("speech_segments", []),
                "audio_events": audio_events,
                "music_features": music_analysis.get("music_features", {})
            }
            
            all_events = []
            
            for segment in speech_analysis.get("speech_segments", []):
                all_events.append({
                    "time": segment["start"],
                    "type": "speech_start",
                    "data": segment
                })
                all_events.append({
                    "time": segment["end"],
                    "type": "speech_end",
                    "data": segment
                })
            
            for event in audio_events:
                all_events.append({
                    "time": event["start_time"],
                    "type": "audio_event",
                    "data": event
                })
            
            all_events.sort(key=lambda x: x["time"])
            
            aligned["timeline"] = all_events
            return aligned
            
        except Exception as e:
            print(f"Timeline alignment error: {e}")
            return {"timeline": [], "speech_segments": [], "audio_events": [], "music_features": {}}
    
    def _create_structured_data(self, 
                               preprocessed_audio: str,
                               separated_sources: Dict[str, str],
                               aligned_results: Dict) -> Dict:
        try:
            return {
                "audio_info": {
                    "preprocessed_path": preprocessed_audio,
                    "separated_sources": separated_sources,
                    "duration": self._get_audio_duration(preprocessed_audio)
                },
                "speech_analysis": {
                    "transcription": aligned_results.get("speech_analysis", {}).get("processed_text", ""),
                    "language": aligned_results.get("speech_analysis", {}).get("language", "en"),
                    "confidence": aligned_results.get("speech_analysis", {}).get("confidence", 0),
                    "emotion": aligned_results.get("speech_analysis", {}).get("emotion", {}),
                    "segments": aligned_results.get("speech_segments", [])
                },
                "music_analysis": {
                    "copyright_status": aligned_results.get("music_analysis", {}).get("copyright_status", {}),
                    "music_features": aligned_results.get("music_features", {}),
                    "detection_confidence": aligned_results.get("music_analysis", {}).get("detection_confidence", 0)
                },
                "audio_events": aligned_results.get("audio_events", []),
                "timeline": aligned_results.get("timeline", [])
            }
            
        except Exception as e:
            print(f"Structured data creation error: {e}")
            return {}
    
    def _generate_summary(self, structured_data: Dict) -> Dict:
        try:
            speech_analysis = structured_data.get("speech_analysis", {})
            music_analysis = structured_data.get("music_analysis", {})
            audio_events = structured_data.get("audio_events", [])
            
            speech_duration = sum(
                seg["end"] - seg["start"] 
                for seg in speech_analysis.get("segments", []) 
                if seg.get("speech", False)
            )
            
            event_counts = {}
            for event in audio_events:
                event_type = event.get("type", "unknown")
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            return {
                "total_duration": structured_data.get("audio_info", {}).get("duration", 0),
                "speech_duration": speech_duration,
                "speech_percentage": (speech_duration / structured_data.get("audio_info", {}).get("duration", 1)) * 100,
                "event_summary": event_counts,
                "copyright_warning": music_analysis.get("copyright_status", {}).get("warning", ""),
                "emotion_summary": speech_analysis.get("emotion", {}).get("emotion", "neutral"),
                "language": speech_analysis.get("language", "en")
            }
            
        except Exception as e:
            print(f"Summary generation error: {e}")
            return {}

    def _get_audio_duration(self, audio_path: str) -> float:
        try:
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except Exception as e:
            print(f"Audio duration calculation error: {e}")
            return 0.0

class AdvancedAudioProcessor:
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.separator = AudioSeparator(self.config)
        self.speech_analyzer = SpeechAnalyzer(self.config)
        self.music_analyzer = MusicAnalyzer(self.config)
        self.event_detector = EventDetector(self.config)
        self.aggregator = AudioAggregator(self.config)
        
        print(f"AdvancedAudioProcessor initialized on device: {self.device}")
    
    def process_audio_for_video(self, video_path: str, output_dir: str) -> Dict[str, Any]:
        import time
        start_time = time.time()
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        try:
            audio_path = os.path.join(output_dir, f"{video_name}.wav")
            self.extract_audio(video_path, audio_path)
            separated_sources = self.separator.separate_audio_sources(audio_path, output_dir)
            vocals_path = separated_sources.get('vocals', None)
            non_vocals_path = separated_sources.get('non_vocals', None)
            if vocals_path and os.path.exists(vocals_path):
                speech_audio = vocals_path
            else:
                speech_audio = audio_path
            speech_analysis = self.speech_analyzer.analyze_speech_content(speech_audio)
            music_input = non_vocals_path if non_vocals_path and os.path.exists(non_vocals_path) else audio_path
            music_analysis = self.music_analyzer.analyze_music_copyright(music_input)
            audio_events = self.event_detector.detect_audio_events(audio_path)
            aggregated_results = self.aggregator.aggregate_results(
                audio_path, separated_sources, speech_analysis, 
                music_analysis, audio_events
            )
            aggregated_results["music_analysis"] = music_analysis
            processing_time = time.time() - start_time
            aggregated_results["metadata"]["processing_time"] = processing_time
            self._save_results(output_dir, video_name, aggregated_results)
            music_title = ""
            music_artist = ""
            shazam_result = music_analysis.get("shazam_result", {})
            if isinstance(shazam_result, dict):
                track = shazam_result.get("track", {})
                music_title = track.get("title", "")
                music_artist = track.get("subtitle", "")
            return {
                "speech_text": speech_analysis.get("processed_text", ""),
                "music_title": music_title,
                "music_artist": music_artist
            }
        except Exception as e:
            print(f"Audio processing error: {e}")
            return {"error": str(e)}

    def extract_audio(self, video_path: str, output_path: str) -> str:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            command = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.config.sample_rate), '-ac', '1', '-y', output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                return output_path
            else:
                print(f"FFmpeg error: {result.stderr}")
                return output_path  
                
        except Exception as e:
            print(f"Audio extraction error: {e}")
            return output_path if output_path is not None else ""

    def _save_results(self, output_dir: str, video_name: str, results: Dict):
        try:
            json_path = os.path.join(output_dir, 'audio_analysis_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            speech_text = None
            if isinstance(results, dict):
                if "speech_text" in results:
                    speech_text = results["speech_text"]
                elif "structured_data" in results and "speech_analysis" in results["structured_data"]:
                    speech_text = results["structured_data"]["speech_analysis"].get("transcription", "")
            txt_path = os.path.join(output_dir, 'speech_transcription.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(speech_text or "")
            print(f"Results saved to: {output_dir}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def has_speech(self, audio_path: str) -> bool:
        speech_analysis = self.speech_analyzer.analyze_speech_content(audio_path)
        return len(speech_analysis.get("speech_segments", [])) > 0
    
    def speech_to_text(self, audio_path: str) -> str:
        speech_analysis = self.speech_analyzer.analyze_speech_content(audio_path)
        return speech_analysis.get("processed_text", "")
    
    def save_speech_text(self, video_output_dir: str, speech_text: str):
        txt_file_path = os.path.join(video_output_dir, 'speech_text.txt')
        try:
            with open(txt_file_path, 'w', encoding='utf-8') as f:
                f.write(speech_text)
            print(f"Speech text saved to: {txt_file_path}")
        except Exception as e:
            print(f"Error saving speech text: {e}")

    def read_speech_text(self, txt_path: str) -> str:
        if not txt_path or not os.path.exists(txt_path):
            return ""
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading speech text: {e}")
            return "" 

class AudioProcessor(AdvancedAudioProcessor):
    pass 