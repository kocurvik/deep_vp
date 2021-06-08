from object_detection.detect_utils import save, show_debug, show_mask_debug, get_bcp_session_filenames
from utils.gpu import set_gpus

import datetime
import os
import argparse
import time

import tensorflow_hub as hub
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max_frames', type=int, default=5000)
    parser.add_argument('--mask', action='store_true', default=False)
    parser.add_argument('-d', '--dump_every', type=int, default=0)
    parser.add_argument('-c', '--conf', type=float, default=0.1)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('path')

    args = parser.parse_args()
    return args


def detect_session(detector, path, session, conf=0.1, dump_every=0, max_frames=5000, mask=False, debug=False):
    print("Starting object detection for ", session)
    if mask:
        json_path = os.path.join(path, 'data', session, 'detections_mask.json')
    else:
        json_path = os.path.join(path, 'data', session, 'detections.json')

    print("Writing to ", json_path)

    filenames = get_bcp_session_filenames(path, session)[:max_frames]

    detection_list = []
    start_time = time.time()

    frame_cnt = 0

    for filename in filenames:
        frame = cv2.imread(os.path.join(path, 'frames', session, filename))
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

        print('{} : {} / {}, ETA: {}'.format(filename, frame_cnt, len(filenames), datetime.timedelta(seconds=(remaining_seconds))))

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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_gpus()

    if args.mask:
        print("Running with mask!")
        object_detector = hub.load("https://hub.tensorflow.google.cn/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1")
    else:
        object_detector = hub.load('https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1')

    path = args.path
    sessions = sorted(os.listdir(os.path.join(path, 'frames')))
    for session in sessions:
        detect_session(object_detector, path, session, conf=args.conf, dump_every=args.dump_every, max_frames=args.max_frames, mask=args.mask, debug=args.debug)


if __name__ == '__main__':
    detect()