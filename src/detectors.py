from typing import List

import numpy as np
from ultralytics import YOLO


class Detector:
    def __init__(self) -> None:
        self.model = None

    def process_images(self, source: np.ndarray, conf_thresh: float = 0.5) -> List:
        """Process images by detection model and return a list of objects

        Args:
            source (np.ndarray): Image numpy array
            conf_thresh (float, optional): Detector confidence threshold.
            Defaults to 0.5.

        Raises:
            NotImplementedError: _description_

        Returns:
            List: [[([x, y, w, h], conf),...],...]
        """
        raise NotImplementedError


class YOLODetector(Detector):
    def __init__(self) -> None:
        super().__init__()
        self.model = YOLO("yolov5n.pt")

    def process_images(self, source, conf_thresh=0.5):
        outputs = self.model.predict(source, conf=conf_thresh, imgsz=640, classes=[0])
        results = []
        for result in outputs:
            boxes = result.numpy().boxes
            for box in boxes:
                results.append((box.xywh[0], box.conf[0]))
        return results
