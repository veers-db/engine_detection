import cv2
import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

"""Extracts top k frame"""
def extract_top_k_extreme_frames(
    video_dir: str, 
    output_dir: str, 
    target_fps_sample: float = 1.0, 
    top_k: int = 10
):
    video_paths = list(Path(video_dir).glob("*.mp4")) #selects all mp4 videos in the specified directory
    os.makedirs(output_dir, exist_ok=True) #creates output directory
    
    all_candidates = [] 
    for vid_path in video_paths:
        cap = cv2.VideoCapture(str(vid_path))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(round(native_fps / target_fps_sample)))
        
        frame_idx = 0
        vid_name = vid_path.stem
        prev_gray_small = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:

                #grayscaling
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                #laplacian filter for eliminating blur frames
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                #resizes the frame
                gray_small = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
                
                #computes structural diff between consecutive frames
                if prev_gray_small is not None:
                    diff_score = np.sum(cv2.absdiff(gray_small, prev_gray_small)) / (gray_small.size * 255.0)
                else:
                    diff_score = 1.0

                #stores all the candidates
                all_candidates.append({
                    'score': diff_score,
                    'frame': frame.copy(),
                    'vid_name': vid_name,
                    'frame_idx': frame_idx,
                    'lap_var': int(lap_var)
                })
                
                prev_gray_small = gray_small
            frame_idx += 1
        cap.release()

    #sorts candidates by diff score and picks top k frames
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    selected_frames = all_candidates[:top_k]
    

    print(f"\nExtracted top {len(selected_frames)} extreme variance frames:")
    for i, item in enumerate(selected_frames):
        out_name = f"{item['vid_name']}_f{item['frame_idx']:06d}_diff{item['score']*100:.1f}%.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, item['frame'], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f" {i+1}. Saved {out_name} (Structural Diff: {item['score']*100:.2f}%)")

if __name__ == "__main__":
    extract_top_k_extreme_frames(
        video_dir=os.getenv("VIDEO_DIR", "videos"),
        output_dir=os.getenv("OUTPUT_DIR", "keyframes"),
        top_k=10
    )