import sys

from utils.gpu import set_gpus

import datetime
import os
import argparse
import json
import time

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', action='store_true', default=False)
    parser.add_argument('-d', '--dump_every', type=int, default=0)
    parser.add_argument('-c', '--conf', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('path')

    args = parser.parse_args()
    return args


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
        # x_min = int(1920 * box[1])
        # y_min = int(1080 * box[0])
        # x_max = min(int(1920 * box[3] + 1), 1920)
        # y_max = min(int(1080 * box[2] + 1), 1080)

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


def get_session_filenames(path, session):
    session_dir = os.path.join(path, 'frames', session)
    print("Checking session dir ", session_dir)

    all_filenames = []

    for dir in sorted(os.listdir(session_dir)):
        dir_path = os.path.join(session_dir, dir)
        print("Checking dir ", dir_path)
        dir_filenames = [os.path.join(dir_path, filename) for filename in sorted(os.listdir(dir_path))]
        all_filenames.extend(dir_filenames)

    return all_filenames


def filter_boxes_scores(boxes, scores, labels, running_mean_frame, frame, conf=0.1):
    filtered_boxes = []
    filtered_scores = []

    diff = np.linalg.norm(frame - running_mean_frame, axis=-1)

    for box, score in zip(boxes, scores):
        x_min = int(1920 * box[1])
        y_min = int(1080 * box[0])
        x_max = int(1920 * box[3] + 1)
        y_max = int(1080 * box[2] + 1)

        diff_box = diff[y_min: y_max, x_min: x_max]
        mean_diff = np.mean(diff_box)
        if mean_diff > 0.1:
            filtered_boxes.append(box.tolist())
            filtered_scores.append(score)

        # print("Mean diff ", mean_diff)
        #
        # box_prev_frame = running_mean_frame[y_min: y_max, x_min: x_max, :]
        # box_frame = frame[y_min: y_max, x_min: x_max, :]
        # cv2.imshow("box running mean", box_prev_frame)
        # cv2.imshow("box", box_frame)
        # cv2.imshow("diff", diff)
        #
        # frame_vis = cv2.rectangle(np.copy(frame), (x_min, y_min), (x_max, y_max), (0, 0, 255))
        # cv2.imshow("Frmae", frame_vis)
        # cv2.waitKey(1)

    return filtered_boxes, filtered_scores



def detect_session(detector, path, session, conf=0.1, dump_every=0, mask=False, debug=False):
    print("Starting object detection for ", session)
    if mask:
        json_path = os.path.join(path, 'data', session, 'detections_mask.json')
    else:
        json_path = os.path.join(path, 'data', session, 'detections.json')

    filenames = get_session_filenames(path, session)


    detection_list = []
    start_time = time.time()

    frame_cnt = 0
    running_mean_frame = np.zeros([1080, 1920, 3])

    for filename in filenames:
        frame = cv2.imread(filename)
        frame_cnt += 1
        if frame is None:
            continue

        result = detector(frame[np.newaxis, :, :, ::-1])

        frame = frame / 255.0

        boxes, labels, scores = result["detection_boxes"].numpy()[0], result["detection_classes"].numpy()[0], \
                                result["detection_scores"].numpy()[0]

        l = np.logical_and(scores > conf, labels == 3)
        boxes = boxes[l]
        scores = scores[l]

        if debug:
            show_debug(np.copy(frame), boxes)

        remaining_seconds = (time.time() - start_time) / frame_cnt * (len(filenames) - frame_cnt)

        print('Frame: {} / {}, ETA: {}'.format(frame_cnt, len(filenames), datetime.timedelta(seconds=(remaining_seconds))))

        item = {'filename':  filename, 'frame_cnt': frame_cnt, 'boxes': boxes.tolist(), 'scores': scores.tolist()}

        if mask:
            masks = result["detection_masks"].numpy()[0][l]
            item['masks'] = masks.tolist()
            if debug:
                show_mask_debug(np.copy(frame), boxes, masks)

        detection_list.append(item)

        if dump_every != 0 and len(detection_list) % dump_every == 0:
            print("Saving at frame ", frame_cnt)
            save(json_path, detection_list)

    print("Saving at frame ", frame_cnt)
    save(json_path, detection_list)
    print("Saved {} bboxes".format(len(detection_list)))


def detect():
    args = parse_args()
    set_gpus()

    if args.mask:
        object_detector = hub.load("https://hub.tensorflow.google.cn/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1")
    else:
        object_detector = hub.load('https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1')
    # object_detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")

    # object_detector = load_model('snapshots/od/resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')

    path = args.path
    sessions = os.listdir(os.path.join(path, 'frames'))
    for session in sessions:
        detect_session(object_detector, path, session, conf=args.conf, dump_every=args.dump_every, mask=args.mask, debug=args.debug)

if __name__ == '__main__':
    detect()