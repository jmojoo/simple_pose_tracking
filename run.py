import argparse

from utils import VideoViewer  # isort: skip
from src import Streamer, Tracker  # isort: skip depends on VideoViewer
from src.detectors import YOLODetector  # isort: skip depends on Streamer


def main(args):
    streamer = Streamer(input_vid=args.input_vid, read_fps=args.read_fps)
    detector = YOLODetector()
    tracker = Tracker(detector, args.max_dist)
    viewer = VideoViewer()
    frame = streamer.get_next_frame()
    while frame is not None:
        tracker.process(frame, **vars(args))
        detections = tracker.tracked_detections
        viewer.add_frame(frame, detections)
        viewer.view_frame()
        frame = streamer.get_next_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-vid", "-i", type=str, required=True, help="Path to input video"
    )
    parser.add_argument(
        "--output-vid", "-o", type=str, required=True, help="Path to output video"
    )
    parser.add_argument(
        "--max-dist",
        "-m",
        type=float,
        default=0.5,
        help="Distance threshold for object tracking",
    )
    parser.add_argument(
        "--min-age", type=int, default=2, help="Minimum age to consider tracked valid"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=2,
        help="Maximum age to consider inactive track lost",
    )
    parser.add_argument(
        "--d-freq",
        "-d",
        type=int,
        default=30,
        help="Run the object detector every <d> frames to discover new objects",
    )
    parser.add_argument("-t", "--tracker", type=str, default="kalman")
    parser.add_argument("--smoothing", type=int, choices=[0, 1], default=0)
    parser.add_argument("--read-fps", "-r", type=float, help="Video reading frame rate")
    args = parser.parse_args()
    main(args)
