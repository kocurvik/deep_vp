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


def filter_boxes(boxes, scores, frame, running_average_frame):
    filtered_boxes = []
    filtered_scores = []

    diff = np.linalg.norm(frame - running_average_frame, axis=-1)

    for box, score in zip(boxes, scores):
        x_min = int(1920 * box[1])
        y_min = int(1080 * box[0])
        x_max = int(1920 * box[3] + 1)
        y_max = int(1080 * box[2] + 1)

        diff_box = diff[y_min: y_max, x_min: x_max]
        mean_diff = np.mean(diff_box)
        if mean_diff > 0.15:
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

    running_average_frame = np.zeros([1080, 1920, 3], dtype=np.float32)

    for detection in detection_data:
        frame_filename = detection['filename']
        frame = cv2.imread(os.path.join(data_path, 'frames', frame_filename))
        frame_cnt = detection['frame_cnt']

        boxes = detection['boxes']
        scores = detection['scores']

        boxes, scores = filter_boxes(boxes, scores, frame, running_average_frame)

        if args.debug:
            show_debug(np.copy(frame), boxes)

        for box, score in zip(boxes, scores):
            box_cnt += 1
            batch_vp_detector.process(frame_cnt, frame, box, score)

            if args.dump_every != 0 and box_cnt % args.dump_every == 0:
                print("Saving at box ", box_cnt)
                save(output_json_path, batch_vp_detector.output_list)

        remaining_seconds = (time.time() - start_time) / (box_cnt + 1) * (total_box_count - box_cnt)
        print('{} : {}, Box: {} / {}, ETA: {}'.format(frame_cnt, frame_filename, box_cnt, total_box_count, datetime.timedelta(seconds=remaining_seconds)))
        running_average_frame = 0.8 * running_average_frame + 0.2 * frame

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