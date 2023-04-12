from ultralytics import YOLO


class Detector:
    def __init__(self) -> None:
        self.model = None

    def process_images(self, source, conf_thresh=0.5):
        raise NotImplementedError


class YOLODetector(Detector):
    def __init__(self) -> None:
        super().__init__()
        self.model = YOLO("yolov8n.pt")

    def process_images(self, source, conf_thresh=0.5):
        outputs = self.model.predict(
            [source, source], conf=conf_thresh, imgsz=640, classes=[0]
        )
        results = []
        for result in outputs:
            dets = []
            boxes = result.numpy().boxes
            for box in boxes:
                dets.append((box.xywh[0], box.conf[0]))
            if len(dets) > 0:
                results.append(dets)
        return results
