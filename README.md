# TikTok Video Feature Extractor

A comprehensive Python toolkit for extracting multimodal features from TikTok videos, including audio, visual, and textual analysis.

## Features

- **Video Processing**: Metadata extraction, keyframe detection, and representative frame selection
- **Audio Analysis**: Speech detection, audio extraction, and transcription using Whisper
- **Visual Analysis**: Object detection (YOLO), scene understanding (BLIP), and blur detection
- **Visual Analysis**: Object detection (YOLO), scene understanding (BLIP)
- **Multimodal Integration**: Combined analysis of audio, visual, and textual features

## Project Structure

```
tiktok/
├── src/
│   ├── audio_processor.py      # Audio extraction and speech recognition
│   ├── video_processor.py      # Video metadata and keyframe extraction
│   ├── frame_analyzer.py       # Frame filtering and representative selection
│   ├── multimodal_extractor.py # YOLO, BLIP, and OCR analysis
│   ├── tiktok_feature_extractor.py # Main controller class
│   └── shared_models.py        # Shared ML models and constants
├── tiktok_videos/              # Input video directory
├── tiktok_frames/              # Output frames directory
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tiktok
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (required for audio extraction):
   - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) or use Chocolatey: `choco install ffmpeg`
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

4. **Download YOLO model** (if not already present):
   ```bash
   # The yolov8x.pt file should be in the project root
   # If missing, it will be downloaded automatically on first use
   ```

## Usage

### Basic Usage

```python
from src.tiktok_feature_extractor import TikTokFeatureExtractor

# Initialize the extractor
extractor = TikTokFeatureExtractor()

# Extract features from a single video
video_path = "tiktok_videos/example.mp4"
output_dir = "tiktok_frames/example"
csv_path = "results.csv"

# Process the video
df = extractor.extract_features_from_single_video(
    video_path=video_path,
    output_folder=output_dir,
    csv_output_path=csv_path
)

# Extract features from all videos in a folder
df = extractor.extract_features_from_folder(
    video_folder="tiktok_videos/",
    output_folder="tiktok_frames/",
    csv_output_path="all_results.csv"
)
```

### Jupyter Notebook

Use the provided Jupyter notebook for interactive analysis:

```bash
jupyter notebook tiktok_video_project.ipynb
```

## Output Features

The extractor generates the following features for each video:

### Video Metadata
- Duration, resolution, frame rate, file size

### Audio Features
- Audio file path
- Speech detection (boolean)
- Transcribed text and length

### Visual Features
- Keyframe count
- Representative frame count
- Representative frame count

### Multimodal Analysis
- YOLO object detection summary
- BLIP scene description summary
- BLIP scene description summary
- Detailed multimodal features JSON

## Class Descriptions

### AudioProcessor
Handles audio extraction, speech detection using Silero VAD, and transcription using OpenAI Whisper.

### VideoProcessor
Extracts video metadata and keyframes using FFmpeg.

### FrameAnalyzer
Filters similar frames, detects blur, and selects representative frames using K-means clustering.

### MultimodalExtractor
Performs object detection (YOLO) and scene understanding (BLIP).

### TikTokFeatureExtractor
Main controller class that orchestrates the entire feature extraction pipeline.

## Troubleshooting

### Audio Extraction Issues
1. Ensure FFmpeg is properly installed and accessible in PATH
2. Check video file has audio stream: `ffprobe video.mp4`
3. Verify audio file creation and size after extraction

### Model Loading Issues
1. Check internet connection for model downloads
2. Ensure sufficient disk space for model files
3. Verify CUDA installation if using GPU

### Memory Issues
1. Process videos one at a time for large datasets
2. Reduce batch sizes in model inference
3. Use CPU instead of GPU if memory is limited

## Requirements

- Python 3.8+
- FFmpeg
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended
- 10GB+ disk space for models and outputs

## License

This project is for research and educational purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request 