import os
import sys

from eval.extract_vp_utils import save
from models.reg import parse_command_line, load_model

import datetime
import argparse
import json
import time

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

from utils.gpu import set_gpus


def show_debug(frame, boxes):
    for box in boxes:
        x_min = int(1920 * box[1])
        y_min = int(1080 * box[0])
        x_max = int(1920 * box[3] + 1)
        y_max = int(1080 * box[2] + 1)

        frame=cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255))

    cv2.imshow("Detections", frame)
    cv2.waitKey(1)


class BatchVPDetector():
    def __init__(self, model, args):
        self.model = model
        self.input_size = args.input_size
        self.batch_size = args.batch_size_eval
        self.debug = args.debug

        self.q = []
        self.output_list = []

    def process(self, frame_cnt, frame, box, score):
        if self.debug:
            self.last_frame = frame

        x_min = int(1920 * box[1])
        y_min = int(1080 * box[0])
        x_max = int(1920 * box[3] + 1)
        y_max = int(1080 * box[2] + 1)

        box_center = np.array([x_min + x_max, y_min + y_max]) / 2
        box_scale = np.array([x_max - x_min, y_max - y_min]) / 2
        try:
            car_img = frame[y_min:y_max, x_min:x_max, :]
            car_img = cv2.resize(car_img, (self.input_size, self.input_size), cv2.INTER_CUBIC)
        except Exception as e:
            print("Caught exception:")
            print(str(e))
            print("Ignoring box")
            return

        vp_box = [y_min, x_min, y_max, x_max]

        item = {'car_img': car_img, 'box': box, 'box_center': box_center, 'box_scale': box_scale, 'vp_box': vp_box, 'frame_cnt': frame_cnt, 'score': score}
        self.q.append(item)
        if len(self.q) == self.batch_size:
            self.predict()

    def finalize(self):
        if len(self.q) > 0:
            self.predict()

    def predict(self):
        cars = [item['car_img'] for item in self.q]
        preds = self.model.predict(np.array(cars) / 255)
        for pred, item in zip(preds, self.q):
            vp1_box = pred[:2]
            vp2_box = pred[2:]

            vp1 = item['box_scale'] * vp1_box + item['box_center']
            vp2 = item['box_scale'] * vp2_box + item['box_center']

            item['box_scale'] = item['box_scale'].tolist()
            item['box_center'] = item['box_center'].tolist()
            item['vp1_box'] = vp1_box.tolist()
            item['vp2_box'] = vp2_box.tolist()
            item['vp1'] = vp1.tolist()
            item['vp2'] = vp2.tolist()
            del item['car_img']

            self.output_list.append(item)

        if self.debug:
            self.draw_debug(item)

        self.q = []

    def draw_debug(self, item):
        x_min = item['vp_box'][1]
        y_min = item['vp_box'][0]
        x_max = item['vp_box'][3]
        y_max = item['vp_box'][2]

        frame = cv2.rectangle(np.copy(self.last_frame), (x_min, y_min), (x_max, y_max), (0, 0, 255))
        try:
            box_center = (int(item['box_center'][0]), int(item['box_center'][1]))
            vp1 = (int(item['vp1'][0]), int(item['vp1'][1]))
            vp2 = (int(item['vp2'][0]), int(item['vp2'][1]))
            frame = cv2.line(frame, vp1, box_center, (0, 255, 0))
            frame = cv2.line(frame, vp2, box_center, (0, 255, 0))
        except Exception:
            ...

        cv2.imshow("VPs debug", frame)
        cv2.waitKey(1)




def detect_session(detector, model_dir_name, data_path, session, args):
    batch_vp_detector = BatchVPDetector(detector, args)

    print("Starting object detection for ", session)
    cap = cv2.VideoCapture(os.path.join(data_path, 'dataset', session, 'video.avi'))
    # DO NOT REMOVE OTHERWISE FRAMES WILL NOT SYNC
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Video loaded!")

    json_path = os.path.join(data_path, 'dataset', session, 'detections.json')
    with open(json_path, 'r') as f:
        detection_data = json.load(f)

    output_json_name = 'VPout_{}_r{}.json'.format(model_dir_name, args.resume)
    output_json_path = os.path.join(data_path, 'dataset', session, output_json_name)

    total_box_count = sum([len(item['boxes']) for item in detection_data])
    print("Loaded {} bounding boxes for {} frames".format(total_box_count, len(detection_data)))
    box_cnt = 0

    start_time = time.time()

    for detection in detection_data:
        for _ in range(args.skip):
            ret, frame = cap.read()

        if not ret or frame is None:
            break

        frame_cnt_orig = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame_cnt = detection['frame_cnt']

        if frame_cnt != frame_cnt_orig:
            raise Exception("Frames from OD do not match frames now! Wrong skip param?")

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
    sessions = sorted(os.listdir(os.path.join(data_path, 'dataset')))
    for session in sessions:
        detect_session(model, model_dir_name, data_path, session, args)

if __name__ == '__main__':
    detect()