import supervision as sv
import os
import cv2

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
        box_annotator = sv.BoxAnnotator()
        image = cv2.imread(data_path)
        with sv.ImageSink(target_dir_path=result_path, overwrite=True) as sink:
            annotated_frame = box_annotator.annotate(scene=image, detections=detections)
            sink.save_image(image=annotated_frame)


def inference_on_video(model, data_path):
    raise NotImplementedError
