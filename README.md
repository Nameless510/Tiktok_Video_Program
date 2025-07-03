# TikTok Video Feature Extractor

A comprehensive Python toolkit for extracting multimodal features from TikTok videos, including audio, visual, and textual analysis.

## Features

- **Video Processing**: Metadata extraction, keyframe detection, and representative frame selection
- **Audio Analysis**: Speech detection, audio extraction, and transcription using Whisper
- **Visual Analysis**: Object detection (YOLO), scene understanding (BLIP), and blur detection
- **Multimodal Integration**: Combined analysis of audio, visual, and textual features
- **AI-Powered Categorization**: Qwen-VL multimodal analysis for product categorization and video description

## Project Structure

```
tiktok/
├── src/
│   ├── audio_processor.py      # Audio extraction and speech recognition
│   ├── video_processor.py      # Video metadata and keyframe extraction
│   ├── frame_analyzer.py       # Frame filtering and representative selection
│   ├── multimodal_extractor.py # YOLO, BLIP, OCR, and Qwen-VL analysis
│   ├── qwen_extractor_example.py # Example script for Qwen-VL usage
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

### Qwen-VL Multimodal Analysis

```python
from src.multimodal_extractor import MultimodalExtractor

# Initialize the extractor
extractor = MultimodalExtractor()

# Example OCR results (from your OCR processing)
ocr_results = {
    "video1_representative_001.jpg": "Lipstick Collection",
    "video1_representative_002.jpg": "Makeup Tutorial"
}

# Example audio transcript (from your audio processing)
audio_transcript = "This is a makeup tutorial showing how to apply lipstick."

# Extract Qwen features with OCR and audio context
qwen_results = extractor.extract_qwen_features(
    keyframes_dir="tiktok_frames/video1",
    video_name="video1",
    ocr_results=ocr_results,
    audio_transcript=audio_transcript
)

# Generate CSV summary with video description and categorization
summary_df = extractor.generate_video_summary_csv(
    qwen_results, 
    "video1_qwen_summary.csv"
)

print(f"Video Description: {summary_df.iloc[0]['Description_of_Video']}")
print(f"Primary Category: {summary_df.iloc[0]['Primary_Category']}")
print(f"Secondary Category: {summary_df.iloc[0]['Secondary_Category']}")
print(f"Tertiary Category: {summary_df.iloc[0]['Tertiary_Category']}")
```

### Using the Example Script

```bash
# Run the Qwen extractor example
python src/qwen_extractor_example.py

# Analyze a single image
python src/qwen_extractor_example.py single

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
- Detailed multimodal features JSON

### AI-Powered Categorization (Qwen-VL)
- **Video Description**: Detailed description of video content
- **Primary Category**: Main product category (e.g., Beauty and Personal Care, Fashion, Electronics)
- **Secondary Category**: Sub-category (e.g., Makeup, Clothing, Mobile Devices)
- **Tertiary Category**: Specific product type (e.g., Foundation, Dresses, Smartphones)
- **Category Confidence**: Confidence scores for each categorization level

The categorization follows TikTok's product hierarchy:
- **Primary**: Beauty and Personal Care, Fashion, Electronics, Home and Garden, Health and Wellness, Food and Beverages, Toys and Entertainment, Sports and Outdoor, Baby and Kids, Pet Supplies
- **Secondary**: Specific subcategories within each primary category
- **Tertiary**: Detailed product types within each subcategory

## Class Descriptions

### AudioProcessor
Handles audio extraction, speech detection using Silero VAD, and transcription using OpenAI Whisper.

### VideoProcessor
Extracts video metadata and keyframes using FFmpeg.

### FrameAnalyzer
Filters similar frames, detects blur, and selects representative frames using CLIP similarity to product-related text prompts.

### MultimodalExtractor
Performs object detection (YOLO), scene understanding (BLIP), and AI-powered categorization using Qwen-VL. Includes methods for:
- `analyze_with_qwen()`: Analyze single image with OCR and audio context
- `extract_qwen_features()`: Extract features from multiple frames
- `generate_video_summary_csv()`: Generate CSV with video description and categorization

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