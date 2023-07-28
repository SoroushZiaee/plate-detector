import argparse
import os

from model.load_model import load_model
from pipe.pipeline import inference_on_image, inference_on_video


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="/Users/soroush/Documents/Code/freelance-project/human-car-detection/car-plate-detection/data/test/test_video_short.mp4",
        help="data path",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="/Users/soroush/Documents/Code/freelance-project/human-car-detection/car-plate-detection/model/ckpt/best.pt",
        help="model path",
    )

    parser.add_argument(
        "--type_detection",
        type=str,
        default="plate",
        help="type detection",
    )

    return parser.parse_args()


def run(data_path: str, model_path: str, type_detection: str):
    filename, extension = os.path.splitext(data_path)

    if extension in [".mp4"]:
        model_plate = load_model(model_path, "plate")
        model_character = load_model(model_path, "character")
        model_type = load_model(model_path, "type")

        inference_on_video(model_plate, model_character, model_type, data_path)

    if extension in [".png", ".jpeg", ".jpg"]:
        model, _ = load_model(model_path, type_detection)
        inference_on_image(model, data_path, type_detection)


def main(conf):
    data_path = conf["data_path"]
    model_path = conf["model_path"]
    type_detection = conf["type_detection"]
    run(data_path, model_path, type_detection)


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    main(conf)
