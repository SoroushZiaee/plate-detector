import cv2
import numpy as np
from math import sqrt, atan, degrees

import supervision as sv
from yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch

from typing import List


def find_longest_line(plate_img_gr):
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(plate_img_gr, (kernel_size, kernel_size), 0)

    low_threshold = 150
    high_threshold = 200

    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(plate_img_gr) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(
        edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )

    lls = []
    for indx, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            line_length = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            lls.append((indx, line_length))
    lls.sort(key=lambda x: x[1])
    linessorted = []
    for indx, ll in lls:
        linessorted.append(lines[indx])
    return linessorted


def find_line_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = degrees(atan(((y2 - y1) / (x2 - x1))))
    return angle


def rotate_image(plate_img_gr, angle):
    h, w, _ = plate_img_gr.shape
    cX, cY = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(plate_img_gr, M, (w, h))
    return rotated


def adjust_cropping(rotated_img):
    h, w, _ = rotated_img.shape
    targ_h = int(w / 4)
    crop_h = int((h - targ_h) / 2)
    cropped_rotated_img = rotated_img[crop_h : h - crop_h, :]
    return cropped_rotated_img


def pad_image(image, height=500, width=500):
    original_height, original_width = image.shape[:2]

    pad_height = max(height - original_height, 0)
    pad_width = max(width - original_width, 0)

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    padded_image = cv2.copyMakeBorder(
        image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT
    )

    return padded_image


def preprocess_image(plate_img_gr):
    # linessorted = find_longest_line(plate_img_gr)
    # rot_angle = find_line_angle(linessorted[-1])
    # rotated_img = rotate_image(plate_img_gr, rot_angle)
    # cropped_rotated_img = adjust_cropping(rotated_img)
    padded_image = pad_image(plate_img_gr)

    return padded_image


def filter(detection, mask: np.ndarray):
    """
    Filter the detections by applying a mask

    :param mask: np.ndarray : A mask of shape (n,) containing a boolean value for each detection indicating if it should be included in the filtered detections
    :param inplace: bool : If True, the original data will be modified and self will be returned.
    :return: Optional[np.ndarray] : A new instance of Detections with the filtered detections, if inplace is set to False. None otherwise.
    """
    detection.xyxy = detection.xyxy[mask]
    detection.confidence = detection.confidence[mask]
    detection.class_id = detection.class_id[mask]
    detection.tracker_id = (
        detection.tracker_id[mask] if detection.tracker_id is not None else None
    )

    return detection


def detections2boxes(detections: sv.Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)


def match_detections_with_tracks(
    detections: sv.Detections, tracks: List[STrack]
) -> sv.Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


# calculate car center by finding center of bbox
def calculate_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def extract_plate_character(detections):
    ALPHABET_IDX_MAPPING = {
        "B": "10",
        "Dal": "11",
        "Ghaf": "12",
        "Gim": "13",
        "H": "14",
        "Lam": "15",
        "Mim": "16",
        "Nun": "17",
        "Sad": "18",
        "Sin": "19",
        "T": "20",
        "Tah": "21",
        "Vav": "22",
        "Ye": "23",
        "plate": "24",
    }

    ENG_TO_PER = {
        "B": "ب",
        "Dal": "د",
        "Ghaf": "ق",
        "Gim": "ل",
        "H": "ه",
        "Lam": "ل",
        "Mim": "م",
        "Nun": "ن",
        "Sad": "ص",
        "Sin": "س",
        "T": "ت",
        "Tah": "ط",
        "Vav": "و",
        "Ye": "ی",
    }

    IDX_ALPHABET_MAPPING = {
        int(value): key for key, value in ALPHABET_IDX_MAPPING.items()
    }

    characters = []
    for idx, (bbox, _, confidence, class_id, _) in enumerate(detections):
        center_x, center_y = calculate_bbox_center(bbox)
        characters.append(
            [
                center_x,
                center_y,
                class_id
                if class_id < 10
                else ENG_TO_PER[IDX_ALPHABET_MAPPING[class_id]],
            ]
        )

    return " ".join(
        list(map(sorted(characters, key=lambda x: x[0], reverse=False), str))
    )
