# TikTok Video Feature Extractor

A comprehensive Python toolkit for extracting multimodal features from TikTok videos, including audio, visual, and textual analysis. Supports both research and production workflows, with modular design and robust test coverage.

---

## Project Overview
This project implements automatic multimodal feature extraction for TikTok short videos. It processes batches of videos to extract:
- Video metadata (duration, resolution, frame rate, file size)
- Audio features (speech, music, events, transcription)
- Visual features (keyframes, representative frames, object detection, blur/black frame filtering)
- Multimodal features (YOLO, BLIP, CLIP, Qwen-VL)
- AI-powered categorization and video description

**Use Cases:**
- Product and content categorization for e-commerce
- Video search and recommendation
- Dataset creation for machine learning
- Research on multimodal video understanding

---

## Workflow
1. **Video Collection**: Place `.mp4` files in `data/tiktok_videos/`.
2. **Metadata Extraction**: Use FFmpeg to extract duration, resolution, frame rate, and file size.
3. **Audio Processing**:
   - Extract audio as 16kHz mono WAV.
   - Separate vocals/non-vocals (Demucs).
   - Transcribe speech (Whisper).
   - Recognize music (Shazamio).
   - Detect events (sound, music, speech, ambient).
   - Save results as JSON, TXT, and CSV.
4. **Keyframe Extraction**:
   - Extract I-frames and sampled frames (1 each second).
   - Extract highlight frames from speech segments.
   - Save all frames in `data/tiktok_frames/<video_name>/`.
5. **Frame Filtering & Selection**:
   - Remove similar frames (SSIM > 0.8).
   - Filter black/blur frames.
   - Detect objects (YOLO-World).
   - Select 2-5 representative frames using CLIP and product prompts.
   - Save representative frames and timestamps.
6. **Multimodal Analysis**:
   - Extract YOLO, BLIP, and CLIP features for each representative frame.
   - Use Qwen-VL-Chat for video description and multi-level categorization.
   - Aggregate results across frames.
7. **Result Saving**:
   - Save all features and analysis to `video_features_results.csv`.
8. **Batch & Single Video Support**: Process single videos or entire folders.

---

## Project Structure
```
tiktok/
├── src/
│   ├── audio_processor.py         # Audio extraction, separation, speech/music/event analysis
│   ├── video_processor.py         # Video metadata and keyframe extraction
│   ├── frame_analyzer.py          # Frame filtering, blur/black detection, representative selection
│   ├── multimodal_extractor.py    # YOLO, BLIP, CLIP, Qwen-VL analysis
│   ├── tiktok_feature_extractor.py# Main pipeline controller
│   └── shared_models.py           # Shared model loading/utilities/constants
├── data/
│   ├── tiktok_videos/             # Input videos
│   └── tiktok_frames/             # Output frames
├── models/                        # Model weights (Qwen, YOLO, etc.)
├── test/                          # Test suite (see below)
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker build
├── docker-compose.yml             # Docker Compose (Jupyter, GPU)
├── tiktok_video_project.ipynb     # Example notebook
├── workflow_description.txt       # Detailed workflow
└── README.md                      # This file
```
---

## Jupyter Notebook
Use the provided notebook for interactive exploration:
jupyter notebook tiktok_video_project.ipynb

---

## Output Features
The extractor generates the following for each video:
- **Video Metadata**: duration, resolution, frame rate, file size
- **Audio**: speech detection, transcription, music recognition, event detection
- **Visual**: keyframe count, representative frames, object detection
- **Multimodal**: YOLO/BLIP/CLIP features, Qwen-VL video description and categorization
- **CSV/JSON/TXT**: All results saved in structured formats

**Example Output Files:**
- `video_features_results.csv`: All features for all videos
- `speech_transcription.txt`: Speech text per video
- `audio_analysis_results.json`: Full audio analysis
- `representative_timestamps.json`: List of selected frames

---

## Module and Class Descriptions
- **audio_processor.py**: Audio extraction, separation (Demucs), speech recognition (Whisper), music recognition (Shazamio), event detection, aggregation.
- **video_processor.py**: Video metadata extraction, keyframe and highlight frame extraction.
- **frame_analyzer.py**: Frame filtering (SSIM, blur, black), YOLO object detection, CLIP-based representative frame selection.
- **multimodal_extractor.py**: Multimodal feature extraction (YOLO, BLIP, CLIP, Qwen-VL), video description, categorization, CSV summary.
- **tiktok_feature_extractor.py**: Main pipeline controller for batch/single video processing.
- **shared_models.py**: Model loading, constants, product/category definitions.

---

## Requirements
- Python 3.8+
- FFmpeg
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended
- 10GB+ disk space for models and outputs
- See `requirements.txt` for all Python dependencies