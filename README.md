# Simple pose tracking
Human pose tracking pipeline, with detection, tracking and pose detection

# Requirements
* Poetry
* Python >= 3.10

# Installation
* Follow instructions at the [official poetry docs](https://python-poetry.org/docs/#installing-with-the-official-installer) to install poetry.
* Clone this repository, `cd` into the simple_pose_tracking repository and run the following command:
```shell
poetry install
```

# Testing
To run the pipeline on a video, run:
```shell
poetry run python run_pipeline.py -i path/to/input/video.mp4 -o path/to/output/video.mp4 --tracker CSRT -r 3
```
This runs the pipeline with CSRT as the tracker, and a frame rate of 3 FPS
