from ultralytics import YOLO
import os

def train_engine_model():
    #using yolo26 small model
    model = YOLO("yolo26s.pt") 

    #Defining the path to our yaml file
    yaml_path = "E:/design-bytes/engine_detection/dataset/dataset.yaml"

    if not os.path.exists(yaml_path):
        print(f"Error: Could not find dataset config at {yaml_path}")
        return

    print("Starting fine-tuning...")

    #Training the model using optimized YOLO26 architecture parameters
    results = model.train(
    data=yaml_path,
    epochs=50,         # 50 iterations for training
    imgsz=640,         # Standard YOLO resolution
    batch=4,           # Small batch size for limited data
    name="engine_v1_small",  # Saves output to runs/detect/engine_v1/
    augment=True,      # Turns on heavy data augmentation to prevent overfitting
    optimizer="auto"   # Allows YOLO26 to auto-select MuSGD / AdamW for optimal stability
    )

    print("\nTraining complete!")
    print(f"Best weights saved to: runs/detect/engine_v1/weights/best.pt")

if __name__ == "__main__":
    train_engine_model()
