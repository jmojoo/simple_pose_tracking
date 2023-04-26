import cv2
import mediapipe as mp
import numpy as np

from src.detection import TrackedDetection

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmark_ds = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4)
conn_ds = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)


def draw_transparent_box(frame, box, color, transparency_factor):
    x1, y1, x2, y2 = box
    white_rect = (np.ones((y2 - y1, x2 - x1, 3), dtype=np.uint8) * color).astype(
        np.uint8
    )
    sub_img = frame[y1:y2, x1:x2]
    res = cv2.addWeighted(sub_img, 0.5, white_rect, transparency_factor, 1.0)
    frame[y1:y2, x1:x2] = res
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
    return frame


class VideoViewer:
    def __init__(self) -> None:
        self.frame_list = []
        self.curr_frame = 0

    def add_frame(self, frame: np.ndarray, detections: dict[int, TrackedDetection]):
        frame = frame.copy()
        for id, detection in detections.items():
            x1, y1, x2, y2 = detection.xyxy
            frame = draw_transparent_box(frame, (x1, y1, x2, y2), (0, 255, 255), 0.5)

            # draw scores
            score_str = "score:{:.4f}".format(detection.conf)
            (w, h), _ = cv2.getTextSize(score_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            frame = cv2.putText(
                frame,
                score_str,
                (x1, y1),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            # Draw track ID
            correction = 5
            track_str = "track ID:{}".format(id)
            (w, h), _ = cv2.getTextSize(track_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            frame = cv2.rectangle(
                frame,
                (x1, y1 + correction),
                (x1 + w, y1 + correction + h),
                (0, 255, 255),
                -1,
            )
            frame = cv2.putText(
                frame,
                track_str,
                (x1, y1 + correction + h),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            # draw pose
            if detection.pose is not None:
                mp_drawing.draw_landmarks(
                    frame,
                    detection.pose,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=landmark_ds,
                    connection_drawing_spec=conn_ds,
                )
        self.frame_list.append(frame)

    def view_frame(self):
        cv2.imshow("visualization", self.frame_list[self.curr_frame])
        cv2.waitKey(0)
        self.curr_frame += 1
