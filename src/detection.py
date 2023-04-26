from typing import Iterable

import mediapipe as mp
import numpy as np

from src.tracking import KalmanFilterTracker, OpenCVTracker, cv_trackers

mp_pose = mp.solutions.pose

XY_GAMMA = 0.95
WH_GAMMA = 0.9
CONF_GAMMA = 0.35


class TrackState:
    def __init__(self, min_age: int, max_age: int) -> None:
        self.active_state = 1
        self.failure_state = 1
        self.min_age = min_age
        self.max_age = max_age

    def register_active(self):
        if self.active_state < self.min_age:
            self.active_state += 1
        if self.failure_state > 1:
            self.failure_state -= 1

    def register_failed(self):
        if self.failure_state < self.max_age:
            self.active_state += 1

    @property
    def tracked(self):
        if self.active_state == self.min_age:
            return True
        else:
            return False

    @property
    def failed(self):
        if self.failure_state == self.max_age:
            return True
        else:
            return False


class TrackedDetection:
    def __init__(
        self, img: np.ndarray, coords: np.ndarray, conf: float, **kwargs
    ) -> None:
        self.x, self.y, self.w, self.h = coords
        self._conf = conf
        self.smoothed = kwargs["smoothing"]
        if kwargs["tracker"] in cv_trackers:
            self.tracker = OpenCVTracker(kwargs["tracker"])
        elif kwargs["tracker"] == "kalman":
            self.tracker = KalmanFilterTracker()
        self.tracker.init(frame=img, box=coords)
        self.pose_estimator = mp_pose.Pose(
            model_complexity=2, min_detection_confidence=0.7
        )
        self._pose = None
        self.state = TrackState(min_age=kwargs["min_age"], max_age=kwargs["max_age"])

    def update_track(self, img: np.ndarray, coords: np.ndarray):
        self.tracker.update(frame=img, box=coords)

    @property
    def xyxy(self):
        x1, y1 = self.x - (self.w / 2), self.y - (self.h / 2)
        x2, y2 = x1 + self.w, y1 + self.h
        return np.array([x1, y1, x2, y2], dtype=np.uint16)

    @property
    def xywh(self):
        return np.array([self.x, self.y, self.w, self.h], dtype=np.uint16)

    @property
    def xywhn(self):
        return self.xywh / self.h

    @property
    def conf(self):
        return self._conf

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, pose):
        self._pose = pose

    @xywh.setter
    def xywh(self, coords: Iterable[int]):
        if self.smoothed:
            # set x, y, w, h with EMA smoothing
            self.x = int(XY_GAMMA * coords[0] + (1 - XY_GAMMA) * self.x)
            self.y = int(XY_GAMMA * coords[1] + (1 - XY_GAMMA) * self.y)
            self.w = int(WH_GAMMA * coords[2] + (1 - WH_GAMMA) * self.w)
            self.h = int(WH_GAMMA * coords[3] + (1 - WH_GAMMA) * self.h)
        else:
            # set x, y, w, h directly
            self.x = int(coords[0])
            self.y = int(coords[1])
            self.w = int(coords[2])
            self.h = int(coords[3])

    @conf.setter
    def conf(self, conf: float):
        if self.smoothed:
            self._conf = int(CONF_GAMMA * conf + (1 - CONF_GAMMA) * self._conf)
        else:
            self._conf = conf

    def predict_track(self, frame: np.ndarray) -> None:
        ret, bbox = self.tracker.predict(frame=frame)
        if ret:
            self.state.register_active()
            return bbox
        else:
            self.state.register_failed()
            return None
