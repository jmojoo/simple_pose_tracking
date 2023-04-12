from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

XY_GAMMA = 0.95
WH_GAMMA = 0.9


class Detection:
    def __init__(self, coords, conf, smoothed=True) -> None:
        self.x, self.y, self.w, self.h = coords
        self.conf = conf
        self.smoothed = smoothed

    @property
    def xyxy(self):
        x1, y1 = self.x - (self.w / 2), self.y - (self.h / 2)
        x2, y2 = x1 + self.w, y1 + self.h
        return np.array([x1, y1, x2, y2], dtype=np.int16)

    @property
    def xywh(self):
        return np.array([self.x, self.y, self.w, self.h], dtype=np.int16)

    @xywh.setter
    def xywh(self, xy: Iterable[int]):
        raise NotImplementedError


class TrackedDetection(Detection):
    def __init__(self, coords, conf) -> None:
        super().__init__(coords, conf)
        self.tracker = cv2.TrackerMOSSE_create()
        self.pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5)

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
