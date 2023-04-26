import cv2
import numpy as np

from utils.label_utils import box_center_to_corner, box_corner_to_center

cv_trackers = {
    "BOOSTING": cv2.legacy.TrackerBoosting_create,
    "MIL": cv2.legacy.TrackerMIL_create,
    "KCF": cv2.TrackerKCF_create,
    "TLD": cv2.legacy.TrackerTLD_create,
    "MEDIANFLOW": cv2.legacy.TrackerMedianFlow_create,
    "GOTURN": cv2.TrackerGOTURN_create,
    "MOSSE": cv2.legacy.TrackerMOSSE_create,
    "CSRT": cv2.TrackerCSRT_create,
}


class BaseTracker:
    def __init__(self, tracker_name):
        if tracker_name in cv_trackers:
            self.tracker = cv_trackers[tracker_name]()

    def init(self, **kwargs):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError


class OpenCVTracker(BaseTracker):
    def __init__(self, tracker_name):
        super().__init__(tracker_name)

    def init(self, **kwargs):
        frame = kwargs["frame"]
        box = box_center_to_corner(kwargs["box"])
        self.tracker.init(frame, box)

    def predict(self, **kwargs):
        frame = kwargs["frame"]
        status, prediction = self.tracker.update(frame)
        if status is not None:
            prediction = box_corner_to_center(prediction)
            return status, prediction
        else:
            return status, None

    def update(self, **kwargs):
        frame = kwargs["frame"]
        box = box_center_to_corner(kwargs["box"])
        self.tracker.init(frame, box)


# class KalmanFilterTracker(BaseTracker):
#     def __init__(self, dim: int = 4):
#         self.kalman = KalmanFilter()
#         self.mean = None
#         self.covariance = None
#         self.dim = dim

#     def init(self, **kwargs):
#         box = kwargs["box"]
#         self.mean, self.covariance = self.kalman.initiate(box)
#         self.dim = len(box)
#         self.update(box=box)

#     def predict(self, **kwargs):
#         self.mean, self.variance = self.kalman.predict(self.mean, self.covariance)
#         return True, self.mean[: self.dim].astype(int)

# def update(self, **kwargs):
#     box = kwargs["box"]
#     self.mean, self.covariance = self.kalman.update(
#         self.mean, self.covariance, box
#         )


class KalmanFilterTracker(BaseTracker):
    def __init__(self):
        self.kalman = cv2.KalmanFilter(6, 6)
        dt = 1 / 5
        self.kalman.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

        self.kalman.transitionMatrix = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            np.float32,
        )

        self.kalman.processNoiseCov = (
            np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1],
                ],
                np.float32,
            )
            * 0.1
        )
        self.vel = [[0], [0]]
        self.init_state = None

    def init(self, **kwargs):
        box = kwargs["box"]
        init_measure = np.array(
            [[box[0]], [box[1]], [box[2]], [box[3]], [0], [0]], np.float32
        )
        self.init_state = init_measure
        # self.update(box=box)
        # print(self.statePre)
        # exit()

    def predict(self, **kwargs):
        pred = self.kalman.predict() + self.init_state
        self.vel = pred[4:]
        return True, np.squeeze(pred[:4])

    def update(self, **kwargs):
        box = kwargs["box"]
        correction = [[box[0]], [box[1]], [box[2]], [box[3]], self.vel[0], self.vel[1]]
        correction = np.array(correction, np.float32) - self.init_state
        self.kalman.correct(correction)
