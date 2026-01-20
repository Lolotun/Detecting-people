import cv2


def draw_detections(frame, detections, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes and labels on frame."""
    for (x1, y1, x2, y2), conf in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"person {conf:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return frame
