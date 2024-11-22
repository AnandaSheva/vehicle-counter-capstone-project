from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  

# Train the model
model.train(
    data="/home/codespace/vehicle-counter-capstone-project/ifdhal/Vehicle-Detection-2/data.yaml",  
    epochs=50,
    imgsz=640
)