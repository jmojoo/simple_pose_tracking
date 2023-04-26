from typing import Union

import cv2


class Streamer:
    def __init__(self, input_vid: str, read_fps: Union[float, None]) -> None:
        self.vid_cap = cv2.VideoCapture(input_vid)
        fps = self.vid_cap.get(cv2.CAP_PROP_FPS)
        if read_fps:
            self.n_skip = max(int(fps / read_fps) - 1, 0)
        else:
            self.n_skip = 0

    def get_next_frame(self):
        for _ in range(self.n_skip):
            self.vid_cap.read()
        success, frame = self.vid_cap.read()
        if not success:
            print("Finished streaming. " "Releasing resources...")
            self.vid_cap.release()
            return None
        else:
            return frame[:, :, ::-1]
