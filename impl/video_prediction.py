import os
import sys
from dotenv import load_dotenv
import supervision as sv
from ultralytics import YOLO

# Load environment configuration from .env if present
load_dotenv()

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from framework.fsm.engine import StateMachine
from impl.states.engine_absent_state import EngineAbsentState

# 1. Configuration & Paths
VIDEO_PATH = os.getenv("VIDEO_PATH", r"E:\design-bytes\engine_detection\record_2026-09-03_12-39.mp4")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", r"runs\detect\engine_fsm_prediction_output.mp4")
MODEL_PATH = os.getenv("MODEL_PATH", r"runs\detect\engine_v1_small\weights\best.pt")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# 2. Initialize Local YOLO Engine
print(f"Loading local YOLO model weights from: {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# 3. Initialize Finite State Machine (10 consecutive frames debounce hysteresis)
fsm = StateMachine(initial_state=EngineAbsentState(), debounce_threshold=10)

# 4. Setup Roboflow Supervision Annotators
box_annotator = sv.BoxAnnotator(
    thickness=3,
    color=sv.Color(r=0, g=255, b=0)
)

label_annotator = sv.LabelAnnotator(
    text_scale=0.7,
    text_thickness=2,
    text_color=sv.Color(r=0, g=0, b=0),
    color=sv.Color(r=0, g=255, b=0),
    text_position=sv.Position.TOP_LEFT
)

# 5. Read video metadata with Supervision
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
print(f"Loaded video: {video_info.width}x{video_info.height} @ {video_info.fps:.1f} fps ({video_info.total_frames} frames)")
print("Engine Detection FSM initialized with 10-frame debounce hysteresis.")


def run_inference(frame):
    """
    Executes inference via local YOLO model,
    returning a standardized sv.Detections object.
    """
    results = model.predict(
        source=frame,
        conf=0.20,
        iou=0.45,
        verbose=False,
        device="cpu"
    )[0]
    detections = sv.Detections.from_ultralytics(results)
    labels = [
        f"{model.names.get(cls_id, 'Engine')}: {confidence:.2f}"
        for confidence, cls_id in zip(detections.confidence, detections.class_id)
    ]
    return detections, labels


# 6. Process frame-by-frame with Supervision VideoSink
frame_count = 0
with sv.VideoSink(target_path=OUTPUT_PATH, video_info=video_info) as sink:
    for frame in sv.get_video_frames_generator(source_path=VIDEO_PATH):
        frame_count += 1

        # Run inference (Local YOLO)
        detections, labels = run_inference(frame)

        # Filter out spurious whole-frame boxes (>90% dimensions)
        if len(detections) > 0:
            box_widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
            box_heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
            valid_mask = (box_widths < 0.90 * video_info.width) & (box_heights < 0.90 * video_info.height)
            detections = detections[valid_mask]
            labels = [label for label, valid in zip(labels, valid_mask) if valid]

        has_engine = len(detections) > 0

        # Step Finite State Machine with current frame detection result
        fsm.step(has_engine=has_engine, raw_detections=detections)

        # Annotate bounding boxes using Supervision
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Write to output video
        sink.write_frame(frame=annotated_frame)

        if frame_count % 50 == 0:
            print(
                f"Processed {frame_count}/{video_info.total_frames} frames "
                f"| FSM Status: {fsm.status} "
                f"(Detections: {fsm.context.consecutive_detections}, Missing: {fsm.context.consecutive_missing})"
            )

print(f"\nProcessing complete! Video saved to: {OUTPUT_PATH}")
print(f"Total State Transitions Recorded: {len(fsm.context.transition_history)}")
for idx, event in enumerate(fsm.context.transition_history, 1):
    print(f"  {idx}. Frame {event.frame_idx}: {event.from_state} -> {event.to_state}")
