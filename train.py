from ultralytics import YOLO

DATASET_PATH = './aadhaar-dataset'

model = YOLO("yolo11n.pt")
model.train(
    data=f"{DATASET_PATH}/data.yaml",
    epochs=5,
    imgsz=640,
    device='cpu' # Tip: use GPU if you've got it.
)
