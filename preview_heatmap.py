import argparse
import os

import cv2
import numpy as np

from tensorflow import keras

import tensorflow_hub as hub

from keras_retinanet.models import load_model as load_od_model

from models.hourglass import create_hourglass_network, heatmap_mean_accuracy, load_model, parse_command_line
from utils.heatmap_dataset import HeatmapBoxCarsDataset
from utils.diamond_space import heatmap_to_vp, process_heatmap, process_heatmap_old, get_focal, process_heatmaps
from utils.video import get_cap


def preview():
    args = parse_command_line()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    scales = [0.03, 0.1, 0.3, 1.0]

    heatmap_model, _, _ = load_model(args, scales)

    print("Heatmap model loaded!")

    object_detecor = hub.load('https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512/1')
    # object_detecor = hub.load('https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_800x1333/1')

    print("Object detection model loaded!")


    cap = cv2.VideoCapture('D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset/session6_left/video.avi')
    # cap = cv2.VideoCapture('D:/Skola/PhD/data/BrnoCarPark/videos/video.mp4')
    # cap = get_cap('D:/Skola/PhD/data/BrnoCarPark/frames/S01/000')
    mask = cv2.imread('D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset/session6_left/video_mask.png', 0)

    pp = np.array([960.5, 540.5])

    back_sub = cv2.createBackgroundSubtractorMOG2()
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))


    vp1s = []
    vp2s = []
    fs = []
    ms = []
    b1s = []
    b2s = []

    ret = True
    while ret:
        for _ in range(10):
            ret, frame = cap.read()
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # frame_od = cv2.resize(frame, (512, 512))
        frame_od = frame
        cv2.imshow("pre-mask frame_od", frame_od)

        # frame_od_mask = back_sub.apply(frame_od)
        # frame_od_mask = cv2.morphologyEx(frame_od_mask, cv2.MORPH_OPEN, kernel_1)
        # frame_od_mask = cv2.morphologyEx(frame_od_mask, cv2.MORPH_DILATE, kernel_2)
        # frame_od = cv2.bitwise_and(frame_od, frame_od, mask=frame_od_mask)

        result = object_detecor(frame_od[np.newaxis, :, :, ::-1])
        boxes, labels, scores = result["detection_boxes"].numpy()[0], result["detection_classes"].numpy()[0], result["detection_scores"].numpy()[0]

        boxes = boxes[np.logical_and(scores > 0.3, labels == 3)]
        cv2.imshow("frame_od", frame_od)
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

            cv2.imshow("car", car)

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

            cv2.imshow("vps", frame_scale)
            cv2.waitKey(1)





if __name__ == '__main__':
    preview()