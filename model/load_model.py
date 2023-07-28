from ultralytics import YOLO
import os


def load_model(model_path: str, type_detection: str):
    if type_detection == "plate":
        model_path = os.path.join(model_path, "plate-detector.pt")
        return YOLO(model_path)

    if type_detection == "character":
        model_path = os.path.join(
            model_path, "character-detector.pt"
        )  # character-detector
        return YOLO(model_path)

    return None
