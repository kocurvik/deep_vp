import sys

from object_detection.detect_utils import show_debug, save, show_mask_debug
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
    parser.add_argument('-s', '--skip', type=int, default=10)
    parser.add_argument('-c', '--conf', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('-m', '--max_frames', type=int, default=0)
    parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('path')

    args = parser.parse_args()
    return args


def detect_session(detector, path, session, max_frames=0, skip=10, conf=0.1, dump_every=0, mask=False, debug=False):
    print("Starting object detection for ", session)

    cap = cv2.VideoCapture(os.path.join(path, 'dataset', session, 'video.avi'))
    video_mask = cv2.imread(os.path.join(path, 'dataset', session, 'video_mask.png'), 0)

    if mask:
        json_path = os.path.join(path, 'dataset', session, 'detections_mask.json')
    else:
        json_path = os.path.join(path, 'dataset', session, 'detections.json')

    print("Writing into ", json_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video has {} frames".format(total_frames))

    if max_frames == 0:
        max_frames = total_frames
    else:
        max_frames = min(max_frames, total_frames)

    print("Processing only the first {} frames".format(max_frames))

    detection_list = []
    start_time = time.time()
    ret = True
    while ret:
        for _ in range(skip):
            ret, frame = cap.read()

        if frame is None:
            break

        frame_cnt = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if frame_cnt > max_frames:
            break

        frame = cv2.bitwise_and(frame, frame, mask=video_mask)

        result = detector(frame[np.newaxis, :, :, ::-1])
        boxes, labels, scores = result["detection_boxes"].numpy()[0], result["detection_classes"].numpy()[0], \
                                result["detection_scores"].numpy()[0]
        l = np.logical_and(scores > conf, labels == 3)
        boxes = boxes[l]
        scores = scores[l]

        if debug:
            show_debug(frame, boxes)

        remaining_seconds = (time.time() - start_time) / frame_cnt * (max_frames - frame_cnt)

        print('Frame: {} / {}, ETA: {}'.format(frame_cnt, max_frames, datetime.timedelta(seconds=(remaining_seconds))))

        item = {'frame_cnt': frame_cnt, 'boxes': boxes.tolist(), 'scores': scores.tolist()}

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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_gpus()

    if args.mask:
        print("Running with mask!")
        object_detector = hub.load("https://hub.tensorflow.google.cn/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1")
    else:
        object_detector = hub.load('https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1')
    # object_detector = hub.load("https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")

    # object_detector = load_model('snapshots/od/resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')

    path = args.path
    sessions = sorted(os.listdir(os.path.join(path, 'dataset')))
    for session in sessions:
        detect_session(object_detector, path, session, max_frames=args.max_frames, skip=args.skip, conf=args.conf, dump_every=args.dump_every, mask=args.mask, debug=args.debug)

if __name__ == '__main__':
    detect()