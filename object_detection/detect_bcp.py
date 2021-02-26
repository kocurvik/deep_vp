import sys

from utils.gpu import set_gpus

sys.path.append('..')

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


def get_session_filenames(path, session):
    session_dir = os.path.join(path, 'frames', session)
    print("Checking session dir ", session_dir)

    filenames = []

    for dir in os.listdir(session_dir):
        dir_path = os.path.join(session_dir, dir)
        print("Checking dir ", dir_path)
        filenames.extend([os.path.join(dir_path, filename) for filename in os.listdir(dir_path)])

    return filenames

def detect_session(detector, path, session, conf=0.1, dump_every=0, debug=False):
    print("Starting object detection for ", session)
    json_path = os.path.join(path, 'data', session, 'detections.json')

    filenames = get_session_filenames(path, session)


    detection_list = []
    start_time = time.time()

    frame_cnt = 0

    for filename in filenames:
        frame = cv2.imread(filename)
        frame_cnt += 1
        if frame is None:
            continue

        result = detector(frame[np.newaxis, :, :, ::-1])
        boxes, labels, scores = result["detection_boxes"].numpy()[0], result["detection_classes"].numpy()[0], \
                                result["detection_scores"].numpy()[0]
        l = np.logical_and(scores > conf, labels == 3)
        boxes = boxes[l]
        scores = scores[l]

        if debug:
            show_debug(frame, boxes)

        remaining_seconds = (time.time() - start_time) / frame_cnt * (len(filenames) - frame_cnt)

        print('Frame: {} / {}, ETA: {}'.format(frame_cnt, len(filenames), datetime.timedelta(seconds=(remaining_seconds))))

        item = {'frame_cnt': frame_cnt, 'boxes': boxes.tolist(), 'scores': scores.tolist()}
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

    object_detector = hub.load('https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1')
    # object_detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")

    # object_detector = load_model('snapshots/od/resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')

    path = args.path
    sessions = os.listdir(os.path.join(path, 'frames'))
    for session in sessions:
        detect_session(object_detector, path, session, conf=args.conf, dump_every=args.dump_every, debug=args.debug)

if __name__ == '__main__':
    detect()