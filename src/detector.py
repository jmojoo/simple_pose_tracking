from ultralytics import YOLO


class Detector:
    def __init__(self) -> None:
        self.model = YOLO("yolov8n.pt")
        self.model.eval()

    def process_images(self, source, conf_thresh=0.5):
        results = self.model.predict(
            source,
        )
        return results
