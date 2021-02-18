import argparse
import os

import tensorflow as tf
from models.hourglass import create_hourglass_network
from tensorflow import keras
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16


def load_model(args):

    print("Initializing model")
    print("Batch size: ", args.batch_size)
    print("Num stacks: ", args.num_stacks)
    print("Input size: {} x {}".format(args.input_size, args.input_size))
    print("Heatmap size: {} x {}".format(args.heatmap_size, args.heatmap_size))
    print("Training for {} epochs".format(args.epochs))
    print("Heatmap feauteres: ", args.features)
    print("Channels: ", args.channels)
    print("Experiment number: ", args.experiment)
    print("Mobilenet version: ", args.mobilenet)

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

    if args.mobilenet:
        module = 'mobilenet'
        module_str = 'm'
    else:
        module = 'bottleneck'
        module_str = 'b'

    snapshot_dir_name = 'VP1VP2_reg_{}_{}_{}in_{}out_{}f_{}n_{}b_{}c_{}'.\
        format(loss_str, module_str, args.input_size, args.heatmap_size, args.features, args.num_stacks, args.batch_size, args.channels, args.experiment)


    backbone = create_hourglass_network(args.features, args.num_stacks, inres=args.input_size, outres=args.heatmap_size,
                                     bottleneck=module, num_channels=args.channels)

    model = keras.models.Sequential()
    model.add(backbone)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(args.features // 2, activation='relu'))
    model.add(keras.layers.Dense(args.features // 4, activation='relu'))
    model.add(keras.layers.Dense(4))

    snapshot_dir_path = os.path.join('snapshots', snapshot_dir_name)

    print("Dir name: ", snapshot_dir_name)

    if args.resume:
        resume_model_path = os.path.join(snapshot_dir_path, 'model.{:03d}.h5'.format(args.resume))
        print("Loading model", resume_model_path)
        model.load_weights(resume_model_path)

    return model, loss, snapshot_dir_name, snapshot_dir_path


def vp1_dist(vp_gt, vp_pred):
    return tf.math.sqrt((vp_gt[:, 0] - vp_pred[:, 0]) ** 2 + (vp_gt[:, 1] - vp_pred[:, 1]) ** 2)


def vp2_dist(vp_gt, vp_pred):
    return tf.math.sqrt((vp_gt[:, 2] - vp_pred[:, 2]) ** 2 + (vp_gt[:, 3] - vp_pred[:, 3]) ** 2)


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
    parser.add_argument('-n', '--num_stacks', type=int, default=2, help='number of stacks')
    parser.add_argument('-i', '--input_size', type=int, default=128, help='size of input')
    parser.add_argument('-o', '--heatmap_size', type=int, default=64, help='size of output heatmaps')
    parser.add_argument('-f', '--features', type=int, default=64, help='number heatmap channels')
    parser.add_argument('-l', '--loss', type=str, default='mse', help='which gpu to use')
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