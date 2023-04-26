from typing import Iterable

import numpy as np


def get_box_from_pose(pose_landmarks, box: Iterable, imsize: Iterable):
    landmarks_norm = np.array([[lm.x, lm.y] for lm in pose_landmarks], dtype=np.float32)
    landmarks = denormalize_pose(landmarks_norm, imsize=box[2:])
    frame_landmarks = box_to_frame_coords(landmarks, box)
    x1, y1 = np.min(frame_landmarks, axis=0)
    x2, y2 = np.max(frame_landmarks, axis=0)
    w, h = int(x2 - x1), int(y2 - y1)
    x, y = int(x1 + (w / 2)), int(y1 + (h / 2))
    return x, y, w, h


def denormalize_pose(pose_coords: np.ndarray, imsize: Iterable):
    pose_coords = pose_coords.copy()
    pose_coords[:, :2] *= imsize
    return pose_coords


def box_to_frame_coords(pose_coords: np.ndarray, box: Iterable):
    pose_coords = pose_coords.copy()
    x, y, w, h = box
    pose_coords[:, 0] += x - (w // 2)
    pose_coords[:, 1] += y - (h // 2)
    return pose_coords


def dilate_box(
    box: Iterable, imsize: Iterable, dilate_factor: tuple[int]
) -> tuple[int]:
    x, y, w, h = box
    imw, imh = imsize
    dw, dh = dilate_factor
    ratios = list(enumerate([w / h, h / w]))
    min_dim, min_ar = min(ratios, key=lambda r: r[1])
    dw *= min_ar if min_dim == 1 else (1 - min_ar)
    dh *= min_ar if min_dim == 0 else (1 - min_ar)
    w *= 1 + dw
    h *= 1 + dh
    x1, y1 = x - (w // 2), y - (h // 2)
    x2, y2 = x1 + w, y1 + h
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(imw, x2), min(imh, y2)
    w, h = x2 - x1, y2 - y1
    x, y = x1 + (w // 2), y1 + (h // 2)
    return int(x), int(y), int(w), int(h)


def box_center_to_corner(box: Iterable) -> tuple[int]:
    x, y, w, h = box
    x = x - (w // 2)
    y = y - (h // 2)
    return x, y, w, h


def box_corner_to_center(box: Iterable) -> tuple[int]:
    x, y, w, h = box
    x = x + (w // 2)
    y = y + (h // 2)
    return x, y, w, h


def box_xywh_to_xyxy(box: Iterable) -> tuple[int]:
    """
    Converts a bounding box from (xc, yc, w, h) format to (x1, y1, x2, y2) format.
    """
    xc, yc, w, h = box
    x1 = xc - w // 2
    y1 = yc - h // 2
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2
