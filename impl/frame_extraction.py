import cv2
import os
import numpy as np
from pathlib import Path
from dotenv import load_file, load_dotenv 

load_dotenv()

def extract_frames(
    video_dir: str, 
    output_dir: str, 
    target_fps_sample: float = 1.0, 
    motion_thresh: float = 2.0
):
    video_paths = list(Path(video_dir).glob("*.mp4"))
    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    prev_gray = None

    for vid_path in video_paths:
        cap = cv2.VideoCapture(str(vid_path))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(native_fps / target_fps_sample)))
        frame_idx = 0
        vid_name = vid_path.stem

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Check for significant difference from previous frame
                if prev_gray is not None:
                    diff = np.mean(cv2.absdiff(gray, prev_gray))
                    if diff < motion_thresh:
                        frame_idx += 1
                        continue

                prev_gray = gray
                out_name = f"{vid_name}_f{frame_idx:06d}.jpg"
                cv2.imwrite(os.path.join(output_dir, out_name), frame)
                saved_count += 1
            
            frame_idx += 1
        
        cap.release()

    print(f"Extraction complete: {saved_count} frames saved to {output_dir}")

if __name__ == "__main__":

    env_video_dir = os.getenv("VIDEO_DIR", "videos")
    env_output_dir = os.getenv("OUTPUT_DIR", "extracted_frames")

    extract_frames(
        video_dir=env_video_dir, 
        output_dir=env_output_dir, 
        target_fps_sample=1.0
    )
