import argparse
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.applications.resnet import ResNet50


def load_model(args):
    model = keras.models.Sequential()
    resnet = ResNet50(include_top=False, weights=None, input_shape=(args.input_size, args.input_size, 3), pooling='avg')
    model.add(resnet)
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(4))

    loss_str = args.loss

    if args.loss == 'mse':
        loss = keras.losses.mse
    elif args.loss == 'mae':
        loss = keras.losses.mae
    elif args.loss == 'rmse':
        loss = keras.metrics.RootMeanSquaredError()
    elif args.loss == 'sl1':
        loss = keras.losses.huber
    else:
        loss, loss_str = get_diamond_loss(args.loss)

    snapshot_dir_name = 'Reg50_{}i_{}_{}'.format(args.input_size, loss_str, args.experiment)
    snapshot_dir_path = os.path.join('snapshots', snapshot_dir_name)

    if args.resume:
        resume_model_path = os.path.join(snapshot_dir_path, 'model.{:03d}.h5'.format(args.resume))
        print("Loading model", resume_model_path)
        model.load_weights(resume_model_path)

    return model, loss, snapshot_dir_name, snapshot_dir_path


def vp1_dist(vp_gt, vp_pred):
    return tf.math.reduce_euclidean_norm(vp_gt[:, :1] - vp_pred[:, :1], axis=-1)


def vp2_dist(vp_gt, vp_pred):
    return tf.math.reduce_euclidean_norm(vp_gt[:, 2:] - vp_pred[:, 2:], axis=-1)


def get_diamond_loss(loss_arg):

    scale = float(loss_arg)
    loss_str = 'diamond{}'.format(scale)

    def _loss(y_pred, y_gt):
        vp1_gt = scale * y_gt[:, :2]
        vp1_pred = scale * y_pred[:, :2]
        vp2_gt = scale * y_gt[:, 2:]
        vp2_pred = scale *y_pred[:, 2:]
        
        vp1_gt_dx = -1 / (tf.sign(vp1_gt[:, 0] * vp1_gt[:, 1]) * vp1_gt[:, 0] + vp1_gt[:, 1] + tf.sign(vp1_gt[:, 1]))
        vp1_gt_dy = vp1_gt[:, 0] / (tf.sign(vp1_gt[:, 0] * vp1_gt[:, 1]) * vp1_gt[:, 0] + vp1_gt[:, 1] + tf.sign(vp1_gt[:, 1]))
        vp1_pred_dx = -1 / (tf.sign(vp1_pred[:, 0] * vp1_pred[:, 1]) * vp1_pred[:, 0] + vp1_pred[:, 1] + tf.sign(vp1_pred[:, 1]))
        vp1_pred_dy = vp1_pred[:, 0] / (tf.sign(vp1_pred[:, 0] * vp1_pred[:, 1]) * vp1_pred[:, 0] + vp1_pred[:, 1] + tf.sign(vp1_pred[:, 1]))
        
        vp2_gt_dx = -1 / (tf.sign(vp2_gt[:, 0] * vp2_gt[:, 1]) * vp2_gt[:, 0] + vp2_gt[:, 1] + tf.sign(vp2_gt[:, 1]))
        vp2_gt_dy =  vp2_gt[:, 0] / (tf.sign(vp2_gt[:, 0] * vp2_gt[:, 1]) * vp2_gt[:, 0] + vp2_gt[:, 1] + tf.sign(vp2_gt[:, 1]))
        vp2_pred_dx = -1 / (tf.sign(vp2_pred[:, 0] * vp2_pred[:, 1]) * vp2_pred[:, 0] + vp2_pred[:, 1] + tf.sign(vp2_pred[:, 1]))
        vp2_pred_dy =  vp2_pred[:, 0] / (tf.sign(vp2_pred[:, 0] * vp2_pred[:, 1]) * vp2_pred[:, 0] + vp2_pred[:, 1] + tf.sign(vp2_pred[:, 1]))

        l = (vp1_gt_dx - vp1_pred_dx) ** 2 + (vp1_gt_dy - vp1_pred_dy) ** 2 + (vp2_gt_dx - vp2_pred_dx) ** 2 + (vp2_gt_dy - vp2_pred_dy) ** 2
        return l

    return _loss, loss_str
    

    

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', type=int, default=0, help='resume from file')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='resume from file')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-i', '--input_size', type=int, default=128, help='size of input')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('-l', '--loss', type=str, default='mse', help='which gpu to use')
    parser.add_argument('--shutdown', action='store_true', default=False, help='shutdown the machine when done')
    parser.add_argument('--half', action='store_true', default=False, help='restrict GPU usage to 8 GB')
    parser.add_argument('-exp', '--experiment', type=int, default=0, help='experiment number')
    parser.add_argument('-w', '--workers', type=int, default=1, help='number of workers for the fit function')
    parser.add_argument('path')
    args = parser.parse_args()
    return args