from ultralytics import YOLO


class Detector:
    def __init__(self, conf_thresh=0.5) -> None:
        self.model = YOLO("yolov8n.pt")
        self.model.eval()
        self.conf_thresh = conf_thresh
