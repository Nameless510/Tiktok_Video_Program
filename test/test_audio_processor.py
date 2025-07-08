"""
Test suite for AdvancedAudioProcessor with modular architecture
"""

import unittest
import os
import tempfile
import numpy as np
import torch
import torchaudio
from unittest.mock import Mock, patch

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from audio_processor import (
    AdvancedAudioProcessor, AudioConfig, 
    AudioPreprocessor, AudioSeparator, SpeechAnalyzer,
    MusicAnalyzer, EventDetector, AudioAggregator
)

class TestAudioConfig(unittest.TestCase):
    
    def test_config_initialization(self):
        """Test AudioConfig initialization."""
        config = AudioConfig()
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.whisper_model_size, "large-v3")
        self.assertEqual(config.language, "en")
        
        # Test custom config
        custom_config = AudioConfig(
            sample_rate=24000,
            language="en",
            vad_threshold=0.7
        )
        self.assertEqual(custom_config.sample_rate, 24000)
        self.assertEqual(custom_config.language, "en")
        self.assertEqual(custom_config.vad_threshold, 0.7)

class TestAudioPreprocessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AudioConfig()
        self.preprocessor = AudioPreprocessor(self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_preprocessor_initialization(self):
        """Test AudioPreprocessor initialization."""
        self.assertIsNotNone(self.preprocessor.config)
        self.assertEqual(self.preprocessor.config.sample_rate, 16000)
        self.assertIn(self.preprocessor.device, ["cuda", "cpu"])
    
    def test_audio_enhancement(self):
        """Test audio enhancement functionality."""
        # Create a test audio file
        test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create audio with noise
        signal = np.sin(2 * np.pi * 440 * t)
        noise = 0.1 * np.random.randn(len(signal))
        noisy_signal = signal + noise
        
        # Save test audio
        torchaudio.save(test_audio_path, torch.tensor(noisy_signal).unsqueeze(0), sample_rate)
        
        # Test enhancement
        enhanced_path = self.preprocessor.enhance_audio_signal(test_audio_path)
        
        self.assertTrue(os.path.exists(enhanced_path))
        self.assertNotEqual(test_audio_path, enhanced_path)
    
    def test_spectral_noise_reduction(self):
        """Test spectral noise reduction."""
        # Create test signal
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Clean signal
        clean_signal = np.sin(2 * np.pi * 440 * t)
        
        # Add noise
        noise = 0.2 * np.random.randn(len(clean_signal))
        noisy_signal = clean_signal + noise
        
        # Apply noise reduction
        denoised_signal = self.preprocessor._spectral_noise_reduction(noisy_signal, sr)
        
        # Check that output has same length
        self.assertEqual(len(denoised_signal), len(noisy_signal))
        self.assertIsInstance(denoised_signal, np.ndarray)
    
    def test_high_pass_filter(self):
        """Test high-pass filter application."""
        # Create test signal with low frequency component
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Signal with both low and high frequency components
        low_freq = np.sin(2 * np.pi * 50 * t)  # 50 Hz
        high_freq = np.sin(2 * np.pi * 1000 * t)  # 1000 Hz
        signal = low_freq + high_freq
        
        # Apply high-pass filter
        filtered_signal = self.preprocessor._apply_high_pass_filter(signal, sr, cutoff=100.0)
        
        # Check output
        self.assertEqual(len(filtered_signal), len(signal))
        self.assertIsInstance(filtered_signal, np.ndarray)
    
    def test_compression(self):
        """Test dynamic range compression."""
        # Create test signal with varying amplitude
        duration = 2.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        
        # Signal with peaks
        signal = np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
        
        # Apply compression
        compressed_signal = self.preprocessor._apply_compression(signal, threshold=0.5, ratio=2.0)
        
        # Check output
        self.assertEqual(len(compressed_signal), len(signal))
        self.assertIsInstance(compressed_signal, np.ndarray)
        
        # Check that peaks are reduced
        max_original = np.max(np.abs(signal))
        max_compressed = np.max(np.abs(compressed_signal))
        
        # Compressed signal should have lower peaks
        self.assertLessEqual(max_compressed, max_original)
    
    def test_spectral_subtraction(self):
        """Test spectral subtraction."""
        # Create test signal
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Clean signal
        clean_signal = np.sin(2 * np.pi * 440 * t)
        
        # Add noise
        noise = 0.15 * np.random.randn(len(clean_signal))
        noisy_signal = clean_signal + noise
        
        # Apply spectral subtraction
        denoised_signal = self.preprocessor._spectral_subtraction(noisy_signal, sr)
        
        # Check output
        self.assertEqual(len(denoised_signal), len(noisy_signal))
        self.assertIsInstance(denoised_signal, np.ndarray)

class TestAudioSeparator(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AudioConfig()
        self.separator = AudioSeparator(self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_separator_initialization(self):
        """Test AudioSeparator initialization."""
        self.assertIsNotNone(self.separator.config)
        self.assertIn(self.separator.device, ["cuda", "cpu"])
    
    @patch('torch.hub.load')
    def test_demucs_model_loading(self, mock_load):
        """Test Demucs model loading."""
        # Mock the demucs model
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        model = self.separator.get_demucs_model()
        
        # Should call torch.hub.load
        mock_load.assert_called_with("facebookresearch/demucs", "demucs", source="github")
        self.assertEqual(model, mock_model)
    
    def test_audio_separation_fallback(self):
        """Test audio separation fallback when model is not available."""
        # Create test audio file
        test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t)
        torchaudio.save(test_audio_path, torch.tensor(signal).unsqueeze(0), sample_rate)
        
        # Test separation with no model
        self.separator._demucs_model = None
        separated_paths = self.separator.separate_audio_sources(test_audio_path, self.temp_dir)
        
        # Should return fallback paths
        self.assertIn("vocals", separated_paths)
        self.assertIn("music", separated_paths)
        self.assertIn("noise", separated_paths)

class TestSpeechAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AudioConfig()
        self.analyzer = SpeechAnalyzer(self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyzer_initialization(self):
        """Test SpeechAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer.config)
        self.assertIn(self.analyzer.device, ["cuda", "cpu"])
    
    @patch('whisper.load_model')
    def test_whisper_model_loading(self, mock_load):
        """Test Whisper model loading."""
        mock_model = Mock()
        mock_load.return_value = mock_model
        
        model = self.analyzer.get_whisper_model()
        
        mock_load.assert_called_with("large-v3", device=self.analyzer.device)
        self.assertEqual(model, mock_model)
    
    def test_emotion_classification(self):
        """Test emotion classification."""
        # Test different emotion scenarios
        emotions = [
            ("excited", 0.2, 3000, 150, 0.15),
            ("happy", 0.08, 2500, 120, 0.06),
            ("sad", 0.02, 800, 80, 0.02),
            ("angry", 0.12, 2000, 140, 0.09),
            ("neutral", 0.04, 1200, 100, 0.04)
        ]
        
        for expected_emotion, energy, centroid, tempo, energy_val in emotions:
            emotion = self.analyzer._classify_emotion(
                np.array([[-5]]), np.array([[centroid]]), tempo, energy_val
            )
            self.assertEqual(emotion, expected_emotion)
    
    def test_text_post_processing(self):
        """Test text post-processing functions."""
        # Test filler word removal
        test_text = "um you know like I mean basically it's actually really good"
        cleaned_text = self.analyzer._post_process_text(test_text)
        
        # Should remove filler words
        self.assertNotIn("um", cleaned_text.lower())
        self.assertNotIn("like", cleaned_text.lower())
        self.assertNotIn("basically", cleaned_text.lower())
        
        # Test common error correction
        test_text = "its cant wont dont"
        corrected_text = self.analyzer._post_process_text(test_text)
        
        # Should fix contractions
        self.assertIn("it's", corrected_text)
        self.assertIn("can't", corrected_text)
        self.assertIn("won't", corrected_text)
        self.assertIn("don't", corrected_text)
        
        # Test punctuation correction
        test_text = "hello world how are you"
        punctuated_text = self.analyzer._post_process_text(test_text)
        
        # Should add proper punctuation
        self.assertTrue(punctuated_text.endswith('.'))

class TestMusicAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AudioConfig()
        self.analyzer = MusicAnalyzer(self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyzer_initialization(self):
        """Test MusicAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer.config)
        self.assertIn(self.analyzer.device, ["cuda", "cpu"])
    
    def test_music_feature_extraction(self):
        """Test music feature extraction."""
        # Create test audio file
        test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t)
        torchaudio.save(test_audio_path, torch.tensor(signal).unsqueeze(0), sample_rate)
        
        # Extract features
        features = self.analyzer._extract_music_features(test_audio_path)
        
        # Should return valid features
        self.assertIn("tempo", features)
        self.assertIn("duration", features)
        self.assertIn("sample_rate", features)
        self.assertIsInstance(features["tempo"], float)
        self.assertIsInstance(features["duration"], float)
    
    def test_copyright_detection(self):
        """Test copyright detection."""
        # Test with music-like features
        features = {"tempo": 120, "duration": 30}
        
        result = self.analyzer._detect_copyright(features)
        
        # Should return valid copyright status
        self.assertIn("status", result)
        self.assertIn("confidence", result)
        self.assertIn("detection_method", result)
        self.assertIn("warning", result)

class TestEventDetector(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AudioConfig()
        self.detector = EventDetector(self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_detector_initialization(self):
        """Test EventDetector initialization."""
        self.assertIsNotNone(self.detector.config)
        self.assertIn(self.detector.device, ["cuda", "cpu"])
    
    def test_sound_event_classification(self):
        """Test sound event classification."""
        # Test different sound event types
        events = [
            ("high_frequency_impact", 0.15, 3500),
            ("medium_frequency_impact", 0.12, 2000),
            ("low_frequency_impact", 0.11, 800),
            ("subtle_sound", 0.05, 1000)
        ]
        
        for expected_type, energy, centroid in events:
            event_type = self.detector._classify_sound_event(energy, centroid)
            self.assertEqual(event_type, expected_type)
    
    def test_audio_event_detection(self):
        """Test audio event detection."""
        # Create test audio file
        test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create audio with some events
        signal = np.sin(2 * np.pi * 440 * t)
        # Add some spikes for event detection
        signal[int(1 * sample_rate)] = 0.5
        signal[int(2 * sample_rate)] = 0.5
        
        torchaudio.save(test_audio_path, torch.tensor(signal).unsqueeze(0), sample_rate)
        
        # Detect events
        events = self.detector.detect_audio_events(test_audio_path)
        
        # Should return a list of events
        self.assertIsInstance(events, list)
        
        # Each event should have required fields
        for event in events:
            self.assertIn("type", event)
            self.assertIn("subtype", event)
            self.assertIn("start_time", event)
            self.assertIn("end_time", event)
            self.assertIn("confidence", event)

class TestAudioAggregator(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AudioConfig()
        self.aggregator = AudioAggregator(self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_aggregator_initialization(self):
        """Test AudioAggregator initialization."""
        self.assertIsNotNone(self.aggregator.config)
    
    def test_timeline_alignment(self):
        """Test timeline alignment."""
        # Mock data
        speech_analysis = {
            "speech_segments": [
                {"start": 0.0, "end": 2.0, "speech": True},
                {"start": 5.0, "end": 7.0, "speech": True}
            ]
        }
        music_analysis = {"music_features": {"tempo": 120}}
        audio_events = [
            {"type": "sound_effect", "start_time": 1.0, "end_time": 1.5},
            {"type": "music", "start_time": 3.0, "end_time": 3.1}
        ]
        
        aligned = self.aggregator._align_timeline(speech_analysis, music_analysis, audio_events)
        
        # Should return aligned timeline
        self.assertIn("timeline", aligned)
        self.assertIn("speech_segments", aligned)
        self.assertIn("audio_events", aligned)
        self.assertIn("music_features", aligned)
        
        # Timeline should be sorted by time
        timeline = aligned["timeline"]
        if len(timeline) > 1:
            for i in range(len(timeline) - 1):
                self.assertLessEqual(timeline[i]["time"], timeline[i + 1]["time"])
    
    def test_structured_data_creation(self):
        """Test structured data creation."""
        # Mock data
        preprocessed_audio = "test_audio.wav"
        separated_sources = {"vocals": "vocals.wav", "music": "music.wav"}
        aligned_results = {
            "speech_analysis": {"processed_text": "Hello world", "language": "en", "confidence": 0.9},
            "music_analysis": {"copyright_status": {"status": "original"}},
            "speech_segments": [],
            "audio_events": [],
            "music_features": {"tempo": 120}
        }
        
        structured = self.aggregator._create_structured_data(
            preprocessed_audio, separated_sources, aligned_results
        )
        
        # Should return structured data
        self.assertIn("audio_info", structured)
        self.assertIn("speech_analysis", structured)
        self.assertIn("music_analysis", structured)
        self.assertIn("audio_events", structured)
        self.assertIn("timeline", structured)
    
    def test_summary_generation(self):
        """Test summary generation."""
        # Mock structured data
        structured_data = {
            "audio_info": {"duration": 60.0},
            "speech_analysis": {
                "segments": [
                    {"start": 0.0, "end": 10.0, "speech": True},
                    {"start": 20.0, "end": 30.0, "speech": True}
                ],
                "emotion": {"emotion": "happy"},
                "language": "en"
            },
            "music_analysis": {"copyright_status": {"warning": "No copyright issues"}},
            "audio_events": [
                {"type": "sound_effect"},
                {"type": "music"},
                {"type": "speech"}
            ]
        }
        
        summary = self.aggregator._generate_summary(structured_data)
        
        # Should return summary
        self.assertIn("total_duration", summary)
        self.assertIn("speech_duration", summary)
        self.assertIn("speech_percentage", summary)
        self.assertIn("event_summary", summary)
        self.assertIn("copyright_warning", summary)
        self.assertIn("emotion_summary", summary)
        self.assertIn("language", summary)
        
        # Check calculations
        self.assertEqual(summary["total_duration"], 60.0)
        self.assertEqual(summary["speech_duration"], 20.0)  # 10 + 10 seconds
        self.assertAlmostEqual(summary["speech_percentage"], 33.33, places=1)

class TestAdvancedAudioProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = AudioConfig()
        self.processor = AdvancedAudioProcessor(self.config)
        
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_processor_initialization(self):
        """Test AdvancedAudioProcessor initialization."""
        self.assertIsNotNone(self.processor.config)
        self.assertIsNotNone(self.processor.preprocessor)
        self.assertIsNotNone(self.processor.separator)
        self.assertIsNotNone(self.processor.speech_analyzer)
        self.assertIsNotNone(self.processor.music_analyzer)
        self.assertIsNotNone(self.processor.event_detector)
        self.assertIsNotNone(self.processor.aggregator)
        self.assertIn(self.processor.device, ["cuda", "cpu"])
    
    def test_audio_extraction(self):
        """Test audio extraction."""
        # This test would require a real video file
        # For now, just test the method exists
        self.assertTrue(hasattr(self.processor, 'extract_audio'))
    
    def test_speech_detection(self):
        """Test speech detection."""
        # Create test audio file
        test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t)
        torchaudio.save(test_audio_path, torch.tensor(signal).unsqueeze(0), sample_rate)
        
        # Test speech detection
        has_speech = self.processor.has_speech(test_audio_path)
        self.assertIsInstance(has_speech, bool)
    
    def test_speech_to_text(self):
        """Test speech to text conversion."""
        # Create test audio file
        test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        signal = np.sin(2 * np.pi * 440 * t)
        torchaudio.save(test_audio_path, torch.tensor(signal).unsqueeze(0), sample_rate)
        
        # Test speech to text
        text = self.processor.speech_to_text(test_audio_path)
        self.assertIsInstance(text, str)
    
    def test_save_and_read_speech_text(self):
        """Test saving and reading speech text."""
        test_text = "Hello world, this is a test."
        
        # Test saving
        self.processor.save_speech_text(self.temp_dir, test_text)
        
        # Test reading
        txt_path = os.path.join(self.temp_dir, 'speech_text.txt')
        read_text = self.processor.read_speech_text(txt_path)
        
        self.assertEqual(read_text, test_text)
    
    def test_save_results(self):
        """Test saving results."""
        results = {
            "structured_data": {
                "speech_analysis": {"transcription": "Test transcription"}
            },
            "summary": {"total_duration": 60.0},
            "metadata": {"processing_time": 5.0}
        }
        
        # Test saving
        self.processor._save_results(self.temp_dir, "test_video", results)
        
        # Check that files were created
        json_path = os.path.join(self.temp_dir, 'audio_analysis_results.json')
        txt_path = os.path.join(self.temp_dir, 'speech_transcription.txt')
        
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(txt_path))

if __name__ == '__main__':
    unittest.main() 