import argparse
import os

import cv2
import numpy as np

from models.load_model import load_model
from tensorflow import keras

from models.hourglass import create_hourglass_network, heatmap_mean_accuracy
from utils.box_cars_dataset import BoxCarsDataset
from utils.diamond_space import heatmap_to_vp


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
    parser.add_argument('path')
    args = parser.parse_args()
    return args

def preview():
    args = parse_command_line()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    scales = [0.03, 0.1, 0.3, 1.0]

    model, _, _ = load_model(args, scales)

    print("Model loaded")

    print("Loading data")

    val_dataset = BoxCarsDataset(args.path, 'val', batch_size=1, img_size=args.input_size,
                                 heatmap_size=args.heatmap_size, scales=scales)

    print("Data loaded")

    for i in range(len(val_dataset)):
        X, y_gt = val_dataset[i]
        y_pred = model.predict(X)

        cv2.imshow("img", X[0])

        for j, scale in enumerate(scales):
            heatmap_gt = y_gt[-1][0, :, :, j + 4]
            heatmap_pred = y_pred[-1][0, :, :, j + 4]

            vp_gt = heatmap_to_vp(heatmap_gt, scale)
            vp_pred = heatmap_to_vp(heatmap_pred, scale)

            print("vp_gt: ", vp_gt)
            print("vp_pred: ", vp_pred)

            cv2.imshow("gt heatmap for scale: {}".format(scale), heatmap_gt)
            cv2.imshow("pred heatmap for scale: {}".format(scale), heatmap_pred)

        cv2.waitKey(0)





if __name__ == '__main__':
    preview()