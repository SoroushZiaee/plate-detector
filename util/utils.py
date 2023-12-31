import cv2
import numpy as np
from math import sqrt, atan, degrees

import supervision as sv
from yolox.tracker.byte_tracker import STrack
from onemetric.cv.utils.iou import box_iou_batch
from farsi_tools import replace_ascii_digits_with_farsi


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
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
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
        "0": "۰",
        "1": "۱",
        "2": "۲",
        "3": "۳",
        "4": "۴",
        "5": "۵",
        "6": "۶",
        "7": "۷",
        "8": "۸",
        "9": "۹",
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

    characters = [
        [
            *calculate_bbox_center(bbox),
            class_id if class_id < 10 else IDX_ALPHABET_MAPPING[class_id],
        ]
        for bbox, _, _, class_id, _ in detections
    ]

    fa_character = [
        [
            *calculate_bbox_center(bbox),
            class_id if class_id < 10 else ENG_TO_PER[IDX_ALPHABET_MAPPING[class_id]],
        ]
        for bbox, _, _, class_id, _ in detections
    ]

    plate_number_list = list(
        map(
            str,
            map(lambda x: x[2], sorted(characters, key=lambda x: x[0], reverse=False)),
        )
    )

    fa_plate_number_list = list(
        map(
            str,
            map(
                lambda x: x[2], sorted(fa_character, key=lambda x: x[0], reverse=False)
            ),
        )
    )

    return plate_number_list, fa_plate_number_list


def is_plate(plate_number_list):
    number_counter = 0
    word_counter = 0

    for char in plate_number_list:
        if char.isdigit():
            number_counter += 1
        else:
            word_counter += 1

    if number_counter == 7 and word_counter == 1:
        return True

    return False


def get_plate_number(plate_number_list):
    # plate_number_list = list(map(replace_ascii_digits_with_farsi, plate_number_list))
    plate_number = f"{str(plate_number_list[0]) + str(plate_number_list[1])} ({plate_number_list[2]}) {str(plate_number_list[3]) + str(plate_number_list[4]) + str(plate_number_list[5])} - {str(plate_number_list[6]) + str(plate_number_list[7])}"

    return plate_number
