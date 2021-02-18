import argparse
import os

import cv2
import numpy as np

from tensorflow import keras

from keras_retinanet.models import load_model as load_od_model

from models.hourglass import create_hourglass_network, heatmap_mean_accuracy, load_model
from utils.heatmap_dataset import HeatmapBoxCarsDataset
from utils.diamond_space import heatmap_to_vp, process_heatmap, process_heatmap_old, get_focal


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', type=int, default=0, help='resume from file')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='resume from file')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-n', '--num_stacks', type=int, default=2, help='number of stacks')
    parser.add_argument('-i', '--input_size', type=int, default=128, help='size of input')
    parser.add_argument('-o', '--heatmap_size', type=int, default=64, help='size of output heatmaps')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('-m', '--mobilenet', action='store_true', default=False)
    parser.add_argument('--shutdown', action='store_true', default=False, help='shutdown the machine when done')
    parser.add_argument('-c', '--channels', type=int, default=256, help='number of channels in network')
    parser.add_argument('-exp', '--experiment', type=int, default=0, help='experiment number')
    parser.add_argument('-w', '--workers', type=int, default=1, help='number of workers for the fit function')
    # parser.add_argument('-s', '--steps', type=int, default=10000, help='steps per epoch')
    # parser.add_argument('path')
    args = parser.parse_args()
    return args

def preview():
    args = parse_command_line()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    scales = [0.03, 0.1, 0.3, 1.0]

    heatmap_model, _, _ = load_model(args, scales)

    print("heatmap model loaded")

    model_od = load_od_model('snapshots/od/resnet50_coco_best_v2.1.0.h5', backbone_name='resnet50')

    cap = cv2.VideoCapture('D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset/session6_left/video.avi')
    # cap = cv2.VideoCapture('D:/Skola/PhD/data/BrnoCarPark/videos/video.mp4')
    mask = cv2.imread('D:/Skola/PhD/data/2016-ITS-BrnoCompSpeed/dataset/session6_left/video_mask.png', 0)

    pp = np.array([960.5, 540.5])


    ret = True
    while ret:
        ret, frame = cap.read()
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        frame_od = cv2.resize(frame, (640, 360))

        boxes, scores, labels = model_od.predict_on_batch(frame_od[np.newaxis, :, :, ::-1])
        boxes = boxes[np.logical_and(scores > 0.5, labels == 2)]
        cv2.imshow("frame_od", frame_od)
        cv2.waitKey(1)

        for box in boxes:
            box = 3 * box
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2] + 1)
            y_max = int(box[3] + 1)

            box_center = np.array([x_min + x_max, y_min + y_max]) / 2
            box_scale = np.array([x_max - x_min, y_max - y_min]) / 2

            car = frame[y_min:y_max, x_min:x_max, :]
            car = cv2.resize(car, (128, 128), cv2.INTER_CUBIC)

            cv2.imshow("car", car)

            heatmap_pred = heatmap_model.predict(car[np.newaxis, ...]/255)

            for j, scale in enumerate(scales):
                heatmap_vp1 = heatmap_pred[-1][0, :, :, j]
                heatmap_vp2 = heatmap_pred[-1][0, :, :, j + 4]

                print("Min vp2 ", np.min(heatmap_vp2))
                print("Max vp2 ", np.max(heatmap_vp2))

                vp1_box, vp1_dist = process_heatmap(heatmap_vp1, scale)
                vp2_box, vp2_dist = process_heatmap(heatmap_vp2, scale)

                # vp1_box[0] *= -1

                vp1 = box_scale * vp1_box + box_center
                vp2 = box_scale * vp2_box + box_center

                focal = get_focal(vp1, vp2, pp)

                print("At scale: {}".format(scale))
                print("vp1: \t {} \t dist: \t {}".format(vp1, vp1_dist))
                print("vp2: \t {} \t dist: \t {}".format(vp2, vp2_dist))
                print("focal length: {}".format(focal))

                cv2.imshow("vp1 heatmap for scale: {}".format(scale), heatmap_vp1/(np.max(heatmap_vp1) + 1e-8))
                cv2.imshow("vp2 heatmap for scale: {}".format(scale), heatmap_vp2/(np.max(heatmap_vp2) + 1e-8))

                frame_scale = np.copy(frame)
                try:
                    frame_scale = cv2.line(frame_scale, (int(box_center[0]), int(box_center[1])), (int(vp1[0]), int(vp1[1])), (0, 255, 0), thickness=2)
                    frame_scale = cv2.line(frame_scale, (int(box_center[0]), int(box_center[1])), (int(vp2[0]), int(vp2[1])), (0, 0, 255), thickness=2)
                except Exception:
                    ...

                cv2.imshow("vps", frame_scale)
                cv2.waitKey(0)





if __name__ == '__main__':
    preview()