from ultralytics import YOLO
from ultralytics import RTDETR
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def load_model(model_name: str = "yolov8n.pt"):
    """Load YOLOv8 model."""
    return YOLO(model_name)


def load_sahi_model(
    model_name: str = "yolov8n.pt",
    model_type: str = "yolov8",
    confidence_threshold: float = 0.27,
    device: str = "cpu"

):
    """
    Load a YOLOv8 model wrapped for SAHI sliced inference.

    Args:
        model_name: Path to the model weights.
        confidence_threshold: Minimum confidence for detections.
        device: Device to run inference on ('cpu' or 'cuda').

    Returns:
        SAHI-compatible detection model.
    """
    return AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_name,
        confidence_threshold=confidence_threshold,
        device=device
    )


def detect_people(
        model,
        frame,
        conf_threshold: float = 0.41,
        imgsz: int = 640):
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


def detect_people_sahi(
    detection_model,
    frame,
    conf_threshold: float = 0.3,
    slice_height: int = 640,
    slice_width: int = 640,
    overlap_ratio: float = 0.2

):
    """
    Run SAHI sliced inference and return only person detections as list of (xyxy, confidence).

    Args:
        detection_model: SAHI AutoDetectionModel instance.
        frame: Input image (numpy array).
        conf_threshold: Minimum confidence to keep detection.
        slice_height, slice_width: Size of each tile.
        overlap_ratio: Overlap between tiles (applies to both height and width).

    Returns:
        List of tuples: [(xyxy, confidence), ...], where xyxy = [x1, y1, x2, y2]
    """
    result = get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_match_threshold=conf_threshold,  # helps reduce duplicates
        verbose=0
    )

    boxes = []
    for obj in result.object_prediction_list:
        if obj.category.name == "person":
            conf = obj.score.value
            x1, y1, x2, y2 = obj.bbox.to_voc_bbox()
            xyxy = [x1, y1, x2, y2]
            boxes.append((xyxy, conf))
    return boxes
