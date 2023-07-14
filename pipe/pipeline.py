import supervision as sv
import os
import cv2


def inference_on_image(model, data_path: str):
    result_path = os.path.join(os.getcwd(), "results")
    os.makedirs(result_path, exist_ok=True)

    results = model(data_path)
    detections = sv.Detections.from_yolov8(results)
    image = cv2.imread(data_path)

    with sv.ImageSink(target_dir_path=result_path, overwrite=True) as sink:
        for xyxy in detections.xyxy:
            cropped_image = sv.crop(image=image, xyxy=xyxy)
            sink.save_image(image=cropped_image)


def inference_on_video(model):
    raise NotImplementedError
