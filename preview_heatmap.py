import os

import cv2
import numpy as np

import tensorflow_hub as hub
from eval.extract_vp_utils import filter_boxes_bcp

from models.hourglass import load_model, parse_command_line
from utils.diamond_space import get_focal, process_heatmaps
from utils.video import get_cap


def preview():
    args = parse_command_line()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    heatmap_model, scales, _, _ = load_model(args)
    print("Heatmap model loaded!")

    object_detecor = hub.load('https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1')
    print("Object detection model loaded!")

    cap = get_cap(args.path)

    vp1s = []
    vp2s = []
    fs = []
    ms = []
    b1s = []
    b2s = []

    prev_edge = None

    ret = True
    while ret:
        # for _ in range(10):
        #     ret, frame = cap.read()
        # frame = cv2.bitwise_and(frame, frame, mask=mask)

        ret, frame = cap.read()

        pp = np.array([frame.shape[1] / 2 + 0.5, frame.shape[0] / 2 + 0.5])

        result = object_detecor(frame[np.newaxis, :, :, ::-1])
        boxes, labels, scores = result["detection_boxes"].numpy()[0], result["detection_classes"].numpy()[0], result["detection_scores"].numpy()[0]
        l = np.logical_and(scores > 0.1, labels == 3)
        boxes = boxes[l]
        scores = scores[l]

        boxes, scores, _, prev_edge = filter_boxes_bcp(boxes, scores, frame, prev_edge)

        # boxes = boxes[np.logical_and(scores > 0.1, labels == 3)]
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

        for box in boxes:
            x_min = int(1920 * box[1])
            y_min = int(1080 * box[0])
            x_max = int(1920 * box[3] + 1)
            y_max = int(1080 * box[2] + 1)

            box_center = np.array([x_min + x_max, y_min + y_max]) / 2
            box_scale = np.array([x_max - x_min, y_max - y_min]) / 2

            car = frame[y_min:y_max, x_min:x_max, :]
            car = cv2.resize(car, (args.input_size, args.input_size), cv2.INTER_CUBIC)

            cv2.imshow("Vehicle", car)

            heatmap_pred = heatmap_model.predict(car[np.newaxis, ...]/255)

            pred_vps, pred_vars = process_heatmaps(heatmap_pred[-1], scales)

            vp1_var = pred_vars[0, :, 0]
            vp2_var = pred_vars[0, :, 1]
            vp1_var_idx = np.argmin(vp1_var, axis=-1)
            vp2_var_idx = np.argmin(vp2_var, axis=-1)

            vp1_box = pred_vps[0, vp1_var_idx, :2]
            vp2_box = pred_vps[0, vp2_var_idx, 2:]

            vp1 = box_scale * vp1_box + box_center
            vp2 = box_scale * vp2_box + box_center

            focal = get_focal(vp1, vp2, pp)
            m = (vp1[1] - vp2[1]) / (vp1[0] - vp2[0])
            b1 = vp1[1] - m * vp1[0]
            b2 = vp2[1] - m * vp2[0]

            if not np.isnan(focal) and not np.isinf(m) and not np.isnan(m):
                vp1s.append(vp1)
                vp2s.append(vp2)
                fs.append(focal)
                ms.append(m)
                b1s.append(b1)
                b2s.append(b2)

                print("VP1: {} \t VP2: {} \t focal: {}".format(vp1, vp2, focal))

                print("Median horizon y = {} * x + {}".format(np.nanmedian(ms), np.nanmedian(np.concatenate([b1s, b2s]))))
                print("Median focal {}".format(np.nanmedian(fs)))

                frame_scale = np.copy(frame)
                try:
                    frame_scale = cv2.line(frame_scale, (int(box_center[0]), int(box_center[1])), (int(vp1[0]), int(vp1[1])), (0, 255, 0), thickness=2)
                    frame_scale = cv2.line(frame_scale, (int(box_center[0]), int(box_center[1])), (int(vp2[0]), int(vp2[1])), (0, 0, 255), thickness=2)
                    frame_scale = cv2.line(frame_scale, (int(vp1[0]), int(vp1[1])), (int(vp2[0]), int(vp2[1])), (0, 255, 255), thickness=2)
                except Exception:
                    ...

                cv2.imshow("Vanishing points", frame_scale)
                cv2.waitKey(1)


if __name__ == '__main__':
    preview()