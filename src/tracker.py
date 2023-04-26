from typing import Dict

import mediapipe as mp
import numpy as np
import scipy.optimize

from src.detection import TrackedDetection
from src.detectors import Detector
from utils.label_utils import box_center_to_corner, dilate_box, get_box_from_pose

mp_pose = mp.solutions.pose
BOX_PRE_DILATE = (1, 1)
BOX_POST_DILATE = (0.7, 0.7)


def _assign_boxes_to_tracks(dists: np.ndarray, max_dist: float):
    dists = dists.copy()
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(dists)
    assignments = []
    unassigned_tracks = []
    unassigned_dets = []
    for r in range(dists.shape[0]):
        if r not in row_idx:
            unassigned_tracks.append(r)
    for c in range(dists.shape[1]):
        if c not in col_idx:
            unassigned_tracks.append(c)
    for r, c in zip(row_idx, col_idx):
        if dists[r, c] > max_dist:
            unassigned_tracks.append(r)
            unassigned_dets.append(c)
        else:
            assignments.append((r, c))
    return assignments, unassigned_tracks, unassigned_dets


class Tracker:
    def __init__(self, detector: Detector, max_distance: float = 0.5) -> None:
        self.tracked_detections: Dict[int, TrackedDetection] = {}
        self.max_dist = max_distance
        self.detector = detector
        self.track_count = 1
        self.frame_count = 0

    def _check_for_new_detections(self, tracked_dets, frame, **kwargs):
        print("\nRunning detector...")
        detections = self.detector.process_images(frame)
        unassigned_dets = []
        if len(tracked_dets) == 0:
            # TODO check if there are stale tracks and assign any detections to them?
            unassigned_dets = list(range(len(detections)))
        if len(detections) > 0 and len(tracked_dets) > 0:
            h = frame.shape[0]
            # normalize bounding box coordinates with img height
            tracked_d = np.array(tracked_dets, dtype=np.float32) / h
            new_d = np.array([d[0] for d in detections], dtype=np.float32) / h
            dists = np.linalg.norm(
                tracked_d[:, np.newaxis] - new_d[np.newaxis], axis=-1
            )
            (
                _,
                _,
                unassigned_dets,
            ) = _assign_boxes_to_tracks(dists, max_dist=self.max_dist)
        for box, conf in [detections[i] for i in unassigned_dets]:
            self.tracked_detections[self.track_count] = TrackedDetection(
                img=frame, coords=box.astype(int), conf=conf, **kwargs
            )
            self.track_count += 1

    def process(self, frame: np.ndarray, **kwargs) -> None:
        tracked_ids = []
        tracked_dets = []
        for id, det in self.tracked_detections.items():
            box = det.predict_track(frame)
            print(box)
            if box is not None:
                tracked_ids.append(id)
                tracked_dets.append(box)

        print("Processing frame...")
        if self.frame_count % kwargs["d_freq"] == 0:
            self._check_for_new_detections(tracked_dets, frame, **kwargs)

        imsize = frame.shape[1::-1]
        for id, box in zip(tracked_ids, tracked_dets):
            self.tracked_detections[id].xywh = box
            # x1, y1, x2, y2 = box_xywh_to_xyxy(box)
            # cv2.imshow("cutout", frame[y1:y2, x1:x2])
            # cv2.waitKey()
            print(f"Tracked:{id}, {box}")
            dilated_box = dilate_box(box, imsize, BOX_PRE_DILATE)
            pose = self.detect_pose(
                frame, self.tracked_detections[id].pose_estimator, dilated_box
            )
            if pose is not None:
                landmarks = pose.landmark

                pose_box = get_box_from_pose(landmarks, dilated_box, imsize)
                # print(box)
                # print(dilated_box)
                # print(new_box)
                # exit()
                post_dilated_box = dilate_box(pose_box, imsize, BOX_POST_DILATE)
                self.tracked_detections[id].xywh = post_dilated_box
                print(f"Updating: {id}, {post_dilated_box}")
                print()
                self.tracked_detections[id].update_track(frame, post_dilated_box)

                # convert in-box pose coordinates to in-image pose coordinates
                # and update box's pose
                x1, y1 = box_center_to_corner(dilated_box)[:2]
                bx_H, bx_W = dilated_box[3], dilated_box[2]
                H_ratio = bx_H / imsize[1]
                W_ratio = bx_W / imsize[0]
                b_Y = y1 / imsize[1]
                b_X = x1 / imsize[0]
                for i in range(len(landmarks)):
                    pose.landmark[i].x = W_ratio * pose.landmark[i].x + b_X
                    pose.landmark[i].y = H_ratio * pose.landmark[i].y + b_Y
                self.tracked_detections[id].pose = pose
            else:
                self.tracked_detections[id].pose = None
        self.frame_count += 1

    def detect_pose(self, frame, pose_estimator: mp_pose.Pose, bbox):
        x, y, w, h = bbox
        x1, y1 = int(x - w / 2), int(y - h / 2)
        # Extract the region of interest (ROI) from the image
        roi = frame[y1 : y1 + int(h), x1 : x1 + int(w)]

        # Process the ROI with mediapipe Pose
        results = pose_estimator.process(roi)

        # Return the pose landmarks for each bounding box
        return results.pose_landmarks
