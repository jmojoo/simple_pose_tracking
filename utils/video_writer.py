import numpy as np

from src.detection import Detection


class VideoWriter:
    def __init__(self) -> None:
        self.frames = []

    def add_frame(self, frame: np.ndarray, tracked_detection: Detection):
        self.frames.append(frame)
