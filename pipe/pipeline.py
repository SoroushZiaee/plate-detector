import supervision as sv
from supervision.draw.color import ColorPalette

# from supervision.video.source import get_video_frames_generator

from yolox.tracker.byte_tracker import BYTETracker

# from supervision.video.sink import VideoSink


import os
import cv2
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from util.utils import (
    preprocess_image,
    filter,
    detections2boxes,
    match_detections_with_tracks,
)


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

        detections = filter(detections, mask)

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
    video_info = sv.VideoInfo.from_video_path(data_path)
    print("\nvideo Info : ", end="")
    print(video_info)

    generator = sv.get_video_frames_generator(data_path)
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.4)

    result_path = os.path.join(os.getcwd(), "video_inference")
    os.makedirs(result_path, exist_ok=True)
    result_path = os.path.join(result_path, "target_video.mp4")

    with sv.VideoSink(target_path=result_path, video_info=video_info) as sink:
        print(f"{video_info.total_frames = }")

        for idx, frame in enumerate(tqdm(generator, total=video_info.total_frames)):
            if idx == 400:
                break

            results = model(frame)
            detections = sv.Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            )

            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )

            tracker_id = match_detections_with_tracks(
                detections=detections, tracks=tracks
            )

            detections.tracker_id = np.array(tracker_id)

            mask = np.array(
                [tracker_id is not None for tracker_id in detections.tracker_id],
                dtype=bool,
            )

            detections = filter(detections, mask)

            for item in detections:
                print(f"{len(item) = }")

            labels = [
                f"#{tracker_id} {confidence:0.2f}"
                for _, confidence, class_id, tracker_id in detections
            ]

            frame = box_annotator.annotate(
                frame=frame, detections=detections, labels=labels
            )

            sink.write_frame(frame)

    raise NotImplementedError
