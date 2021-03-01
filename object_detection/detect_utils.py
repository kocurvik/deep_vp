import json

import cv2
import numpy as np


def save(json_path, detection_list):
    with open(json_path, 'w') as f:
        json.dump(detection_list, f)


def show_debug(frame, boxes):
    for box in boxes:
        x_min = int(1920 * box[1])
        y_min = int(1080 * box[0])
        x_max = int(1920 * box[3] + 1)
        y_max = int(1080 * box[2] + 1)

        frame=cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255))

    cv2.imshow("Detections", frame)
    cv2.waitKey(1)


def show_mask_debug(frame, boxes, masks):
    for box, mask in zip(boxes, masks):
        rect_src = np.array([[0, 0], [mask.shape[1], 0], [mask.shape[1], mask.shape[0]], [0, mask.shape[0]]], dtype=np.float32)
        rect_dst = np.array([[1920 * box[1], 1080 * box[0]], [1920 * box[3], 1080 * box[0]], [1920 * box[3], 1080 * box[2]], [1920 * box[1], 1080 * box[2]]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect_src[:, :], rect_dst[:, :])

        mask_frame = cv2.warpPerspective(mask, M, (1920, 1080), flags=cv2.INTER_CUBIC)
        vis_frame = np.copy(frame)
        vis_frame[:, :, 2] = mask_frame

        cv2.imshow("Masks", vis_frame)
        cv2.imshow("Masks transformed", mask_frame)
        cv2.imshow("Masks original", mask)
        cv2.waitKey(0)