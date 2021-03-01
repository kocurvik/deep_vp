import os

from eval.extract_vp_batch_detector import BatchVPDetector
from models.reg import parse_command_line, load_model

import datetime
import json
import time

import cv2
import numpy as np

from utils.gpu import set_gpus

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


def detect_session(detector, model_dir_name, data_path, session, args):
    batch_vp_detector = BatchVPDetector(detector, args)

    print("Starting vp detection for ", session)
    filenames = get_session_filenames(data_path, session)

    json_path = os.path.join(data_path, 'data', session, 'detections.json')
    with open(json_path, 'r') as f:
        detection_data = json.load(f)

    output_json_name = 'VPout_{}_r{}.json'.format(model_dir_name, args.resume)
    output_json_path = os.path.join(data_path, 'data', session, output_json_name)

    total_box_count = sum([len(item['boxes']) for item in detection_data])
    print("Loaded {} bounding boxes for {} frames".format(total_box_count, len(detection_data)))
    box_cnt = 0

    start_time = time.time()

    for detection, frame_filename in zip(detection_data, filenames):
        frame = cv2.imread(os.path.join(data_path, 'frames', frame_filename))

        frame_cnt = detection['frame_cnt']

        boxes = detection['boxes']
        scores = detection['scores']

        if args.debug:
            show_debug(np.copy(frame), boxes)

        for box, score in zip(boxes, scores):
            box_cnt += 1
            batch_vp_detector.process(frame_cnt, frame, box, score)

            if args.dump_every != 0 and box_cnt % args.dump_every == 0:
                print("Saving at box ", box_cnt)
                save(output_json_path, batch_vp_detector.output_list)

        remaining_seconds = (time.time() - start_time) / (box_cnt + 1) * (total_box_count - box_cnt)
        print('Frame {}, Box: {} / {}, ETA: {}'.format(frame_cnt, box_cnt, total_box_count, datetime.timedelta(seconds=remaining_seconds)))

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