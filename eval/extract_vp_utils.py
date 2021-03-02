import json

import cv2
import numpy as np


class BatchVPDetector():
    def __init__(self, model, args):
        self.model = model
        self.input_size = args.input_size
        self.batch_size = args.batch_size_eval
        self.debug = args.debug

        self.q = []
        self.output_list = []

    def process(self, frame, box, score, **kwargs):
        if self.debug:
            self.last_frame = frame

        x_min = int(frame.shape[1] * box[1])
        y_min = int(frame.shape[0] * box[0])
        x_max = int(frame.shape[1] * box[3] + 1)
        y_max = int(frame.shape[0] * box[2] + 1)

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

        item = {'car_img': car_img, 'box': box, 'box_center': box_center, 'box_scale': box_scale, 'vp_box': vp_box, 'score': score, **kwargs}
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


def save(json_path, detection_list):
    with open(json_path, 'w') as f:
        json.dump(detection_list, f)