import cv2


def get_video_properties(cap):
    """Return width, height, fps, total_frames from VideoCapture."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, total_frames


def create_video_writer(output_path, fps, width, height):
    """Create VideoWriter object for MP4 output."""
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
