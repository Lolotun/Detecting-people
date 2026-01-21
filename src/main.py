import cv2
import os
from .video_io import get_video_properties, create_video_writer
from .detection import load_sahi_model, detect_people_sahi
from .visualization import draw_detections
import subprocess
import torch.cuda


def main():
    input_path = "crowd_5s.mp4"
    output_video_path = "temp_output.avi"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    width, height, fps, _ = get_video_properties(cap)
    out = create_video_writer(output_video_path, fps, width, height)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model =load_sahi_model("yolov8n.pt",0.3, device)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_people_sahi(model, frame, conf_threshold=0.3)
        annotated_frame = draw_detections(frame, detections)
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()

    out.release()
    final_output = "detect.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", output_video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            final_output
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    os.remove(output_video_path)


if __name__ == "__main__":
    main()
