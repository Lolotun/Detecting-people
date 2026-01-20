from ultralytics import YOLO


def load_model(model_name: str = "yolov8n.pt"):
    """Load YOLOv8 model."""
    return YOLO(model_name)


def detect_people(model, frame, conf_threshold: float = 0.3, imgsz: int = 640):
    """Run inference and return only person detections (class 0)."""
    results = model(frame, imgsz=imgsz, verbose=False)
    boxes = []
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id == 0 and conf >= conf_threshold:
            xyxy = box.xyxy[0].cpu().numpy()
            boxes.append((xyxy, conf))
    return boxes
