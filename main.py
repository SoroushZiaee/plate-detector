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
        help="data path",
    )

    return parser.parse_args()


def run(data_path: str, model_path: str):
    filename, extension = os.path.splitext(data_path)
    model = load_model(model_path)

    if extension in [".mp4"]:
        pass

    if extension in [".png", ".jpeg", ".jpg"]:
        inference_on_image(model)


def main(conf):
    data_path = conf["data_path"]
    model_path = conf["model_path"]
    run(data_path)


if __name__ == "__main__":
    opts = parse_args()
    conf = vars(opts)
    main(conf)
