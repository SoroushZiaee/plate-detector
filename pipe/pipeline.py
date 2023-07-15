import supervision as sv

# from supervision.video.dataclasses import VideoInfo
# from supervision.video.source import get_video_frames_generator

# from yolox.tracker.byte_tracker import BYTETracker
# from supervision.video.sink import VideoSink


import os
import cv2
import numpy as np
from dataclasses import dataclass

from util.utils import preprocess_image


def inference_on_image(model, data_path: str, type_detection: str = "plate"):
    if type_detection.lower() == "plate":
        result_path = os.path.join(os.getcwd(), "plates")
        os.makedirs(result_path, exist_ok=True)

        results = model(data_path)[0]
        detections = sv.Detections.from_yolov8(results)
        image = cv2.imread(data_path)

        with sv.ImageSink(target_dir_path=result_path, overwrite=True) as sink:
            for xyxy in detections.xyxy:
                cropped_image = sv.crop(image=image, xyxy=xyxy)
                preprocessed_image = preprocess_image(cropped_image)
                sink.save_image(image=preprocessed_image)

    if type_detection.lower() == "character":
        result_path = os.path.join(os.getcwd(), "characters")
        os.makedirs(result_path, exist_ok=True)
        results = model(data_path)[0]
        detections = sv.Detections.from_yolov8(results)

        mask = np.array(
            [class_id != 24 for class_id in detections.class_id], dtype=bool
        )  # Remove plate detection

        detections.filter(mask=mask, inplace=True)

        box_annotator = sv.BoxAnnotator()
        image = cv2.imread(data_path)
        with sv.ImageSink(target_dir_path=result_path, overwrite=True) as sink:
            annotated_frame = box_annotator.annotate(scene=image, detections=detections)
            sink.save_image(image=annotated_frame)


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def inference_on_video(model, data_path):
    byte_tracker = BYTETracker(BYTETrackerArgs())
    video_info = VideoInfo.from_video_path(data_path)
    print("\nvideo Info : ", end="")
    print(video_info)

    generator = get_video_frames_generator(conf["video_target_path"])

    raise NotImplementedError
