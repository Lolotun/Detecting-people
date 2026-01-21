import cv2
import os
import argparse
from .video_io import get_video_properties, create_video_writer
from .detection import load_model, load_sahi_model,detect_people,  detect_people_sahi
from .visualization import draw_detections
import subprocess
import torch.cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], required=True,
                        help="1: YOLOv8, 2: SAHI+YOLOv8, 3: SAHI+RT-DETR")
    args = parser.parse_args()
    input_path = "crowd_5s.mp4"
    output_video_path = "temp_output.avi"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(args.video)
    width, height, fps, _ = get_video_properties(cap)
    out = create_video_writer(output_video_path, fps, width, height)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    if args.mode == 1:
        model =load_model("yolov8n.pt")
        use_sahi = False
    elif args.mode == 2:
        model = load_sahi_model("yolov8n.pt", confidence_threshold=0.3, device=device)
        use_sahi = True
    elif args.mode == 3:
        model = load_sahi_model("rtdetr-l.pt", confidence_threshold=0.35, device=device)
        use_sahi = True

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.mode == 1:
            detections = detect_people(model, frame, conf_threshold=0.41, imgsz=1280)
        else:
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
