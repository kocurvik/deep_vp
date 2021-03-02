import os

from eval.extract_vp_utils import BatchVPDetector, save
from models.reg import parse_command_line, load_model

import datetime
import json
import time

import cv2
import numpy as np
from object_detection.detect_utils import show_debug

from utils.gpu import set_gpus


def filter_boxes(boxes, scores, frame, prev_edge):
    filtered_boxes = []
    filtered_scores = []

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(frame, 100, 200)
    if prev_edge is None:
        edge_diff = edge
    else:
        edge_diff = np.abs(edge - prev_edge)

    for box, score in zip(boxes, scores):
        x_min = int(frame.shape[1] * box[1])
        y_min = int(frame.shape[0] * box[0])
        x_max = int(frame.shape[1] * box[3] + 1)
        y_max = int(frame.shape[0] * box[2] + 1)

        edge_box = edge_diff[y_min: y_max, x_min: x_max]
        if np.mean(edge_box) > 5.0:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            if len(filtered_scores) >= 10:
                break

    prev_edge = edge

    return filtered_boxes, filtered_scores, prev_edge


def detect_session(detector, model_dir_name, data_path, session, args):
    batch_vp_detector = BatchVPDetector(detector, args)

    print("Starting vp detection for ", session)

    json_path = os.path.join(data_path, 'data', session, 'detections.json')
    with open(json_path, 'r') as f:
        detection_data = json.load(f)

    output_json_name = 'VPout_{}_r{}.json'.format(model_dir_name, args.resume)
    output_json_path = os.path.join(data_path, 'data', session, output_json_name)

    total_box_count = sum([len(item['boxes']) for item in detection_data])
    print("Loaded {} bounding boxes for {} frames".format(total_box_count, len(detection_data)))
    box_cnt = 0

    start_time = time.time()

    # running_average_frame = np.zeros([1080, 1920, 2], dtype=np.float32)
    prev_edge = None

    for detection in detection_data:
        frame_filename = detection['filename']
        frame = cv2.imread(os.path.join(data_path, 'frames', session, frame_filename))
        frame_cnt = detection['frame_cnt']

        boxes = detection['boxes']
        scores = detection['scores']

        box_cnt += len(boxes)

        boxes, scores, prev_edge = filter_boxes(boxes, scores, frame, prev_edge)

        if args.debug:
            show_debug(np.copy(frame), boxes)

        for box, score in zip(boxes, scores):
            batch_vp_detector.process(frame, box, score, frame_cnt=frame_cnt, frame_filename=frame_filename)

            if args.dump_every != 0 and box_cnt % args.dump_every == 0:
                print("Saving at box ", box_cnt)
                save(output_json_path, batch_vp_detector.output_list)

        remaining_seconds = (time.time() - start_time) / (box_cnt + 1) * (total_box_count - box_cnt)
        print('{} : {}, Box: {} / {}, ETA: {}'.format(frame_cnt, frame_filename, box_cnt, total_box_count, datetime.timedelta(seconds=remaining_seconds)))

    batch_vp_detector.finalize()
    print("Saving at box ", box_cnt)
    save(output_json_path, batch_vp_detector.output_list)
    print("Finished session: {} with {} boxes".format(session, total_box_count))


def detect():
    args = parse_command_line()
    set_gpus()

    model, _, model_dir_name, _ = load_model(args)

    data_path = args.path
    sessions = os.listdir(os.path.join(data_path, 'data'))
    for session in sessions:
        detect_session(model, model_dir_name, data_path, session, args)

if __name__ == '__main__':
    detect()