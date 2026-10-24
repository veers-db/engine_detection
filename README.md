# Engine Detection System

This script extracts frames from videos (`.mp4`) and filters out duplicate or static frames using motion detection before saving them for annotation in Label Studio.

## 🚀 Setup & Installation

1. Install the required libraries:
```bash
pip install opencv-python numpy python-dotenv label-studio
```

2. Create a `.env` file in the project folder and configure your paths:
```env
VIDEO_DIR="path/to/your/dataset"
OUTPUT_DIR="path/to/store/extracted/frames"
```
Setup label studio via PowerShell
```bash
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED="true"
label-studio
```
---

## 💻 How to Use

1. Put your raw videos into your designated video folder.
2. Run the script:
```bash
python frame_extraction.py
```

