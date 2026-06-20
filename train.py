from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="data/data.yaml",
    epochs=15,
    imgsz=640,
    batch=8,
    patience=5,
    project="runs",
    name="smoking_train"
)