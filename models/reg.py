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
    print("Input size: {} x {}".format(args.input_size, args.input_size))

    if not args.resnet:
        print("Using hourglass")
        print("Num stacks: ", args.num_stacks)
        print("Heatmap size: {} x {}".format(args.heatmap_size, args.heatmap_size))
        print("Use diamond coords for output: {}".format(args.diamond))
        print("Scale for vp: {}".format(args.scale))

        print("Heatmap feautures: ", args.features)
        print("Channels: ", args.channels)
        print("Mobilenet version: ", args.mobilenet)

    else:
        print("Using ResNet50")

    print("Experiment number: ", args.experiment)

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


    if not args.resnet:
        model_name = 'reg_diamond' if args.diamond else 'reg_orig'
        snapshot_dir_name = 'VP1VP2_{}_{}_{}_{}in_{}out_{}s_{}f_{}n_{}b_{}c_{}'.\
            format(model_name, loss_str, module_str, args.input_size, args.heatmap_size, args.scale, args.features, args.num_stacks, args.batch_size, args.channels, args.experiment)

        backbone = create_hourglass_network(args.features, args.num_stacks, inres=args.input_size, outres=args.heatmap_size,
                                         bottleneck=module, num_channels=args.channels)

        outputs = []
        for i in range(args.num_stacks):
            backbone_out = backbone.outputs[i]
            x =  keras.layers.GlobalAveragePooling2D(name='mlp_pool_{}'.format(i))(backbone_out)
            x = keras.layers.Dense(args.features // 2, activation='relu', name='mlp_1_{}'.format(i))(x)
            x = keras.layers.Dense(args.features // 4, activation='relu', name='mlp_2_{}'.format(i))(x)
            x = keras.layers.Dense(4, name='mlp_out_{}'.format(i))(x)
            outputs.append(x)

        model = keras.models.Model(inputs=backbone.input, outputs=outputs)

    else:
        if args.num_stacks != 1:
            raise Exception("Cannot use ResNet with multiple outputs!")
        model_name = 'resnet_diamond' if args.diamond else 'resnet_orig'
        snapshot_dir_name = 'VP1VP2_{}_{}_{}in_{}s_{}b_{}'.\
            format(model_name, loss_str, args.input_size,  args.scale, args.batch_size, args.experiment)

        model = keras.models.Sequential()
        backbone = ResNet50(input_shape=(args.input_size, args.input_size, 3), include_top=False, pooling='avg')
        model.add(backbone)
        model.add(keras.layers.Dense(128, activation='relu', name='mlp_1'))
        model.add(keras.layers.Dense(64, activation='relu', name='mlp_2'))
        model.add(keras.layers.Dense(4, name='mlp_out'))

    print("Dir name: ", snapshot_dir_name)
    snapshot_dir_path = os.path.join('snapshots', snapshot_dir_name)

    if args.resume:
        resume_model_path = os.path.join(snapshot_dir_path, 'model.{:03d}.h5'.format(args.resume))
        print("Loading model", resume_model_path)
        model.load_weights(resume_model_path)

    return model, loss, snapshot_dir_name, snapshot_dir_path


def get_metrics(use_diamond=False, scale=1.0):
    if use_diamond:
        def vp1_dist(vp_gt, vp_pred):
            return tf.divide(vp1_diamond_dist(vp_gt, vp_pred), scale)

        def vp2_dist(vp_gt, vp_pred):
            return tf.divide(vp2_diamond_dist(vp_gt, vp_pred), scale)

    else:
        def vp1_dist(vp_gt, vp_pred):
            return tf.divide(tf.math.sqrt((vp_gt[:, 0] - vp_pred[:, 0]) ** 2 + (vp_gt[:, 1] - vp_pred[:, 1]) ** 2), scale)

        def vp2_dist(vp_gt, vp_pred):
            return tf.divide(tf.math.sqrt((vp_gt[:, 2] - vp_pred[:, 2]) ** 2 + (vp_gt[:, 3] - vp_pred[:, 3]) ** 2), scale)

    return [vp1_dist, vp2_dist]


def original_coords_from_diamond_tf(vp):
    vp_d_x = tf.math.divide_no_nan(vp[:, 1], vp[:, 0])
    vp_d_y = tf.math.divide_no_nan(tf.sign(vp[:, 0]) * vp[:, 0] + tf.sign(vp[:, 1]) * vp[:, 1] - 1, vp[:, 0])
    return tf.stack([vp_d_x, vp_d_y], axis=-1)


def vp1_diamond_dist(vp_d_gt, vp_d_pred):
    vp_gt = original_coords_from_diamond_tf(vp_d_gt[:, :2])
    vp_pred = original_coords_from_diamond_tf(vp_d_pred[:, :2])
    return tf.math.sqrt((vp_gt[:, 0] - vp_pred[:, 0]) ** 2 + (vp_gt[:, 1] - vp_pred[:, 1]) ** 2)


def vp2_diamond_dist(vp_d_gt, vp_d_pred):
    vp_gt = original_coords_from_diamond_tf(vp_d_gt[:, 2:])
    vp_pred = original_coords_from_diamond_tf(vp_d_pred[:, 2:])
    return tf.math.sqrt((vp_gt[:, 0] - vp_pred[:, 0]) ** 2 + (vp_gt[:, 1] - vp_pred[:, 1]) ** 2)

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
    parser.add_argument('-n', '--num_stacks', type=int, default=1, help='number of stacks')
    parser.add_argument('-i', '--input_size', type=int, default=128, help='size of input')
    parser.add_argument('-o', '--heatmap_size', type=int, default=64, help='size of output heatmaps')
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='scale to use for vp')
    parser.add_argument('-d', '--diamond', action='store_true', default=False, help='whether to use diamond space for output')
    parser.add_argument('-f', '--features', type=int, default=64, help='number heatmap channels')
    parser.add_argument('-l', '--loss', type=str, default='mse', help='which gpu to use')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('-m', '--mobilenet', action='store_true', default=False)
    parser.add_argument('--resnet', action='store_true', default=False)
    parser.add_argument('--shutdown', action='store_true', default=False, help='shutdown the machine when done')
    parser.add_argument('-c', '--channels', type=int, default=256, help='number of channels in network')
    parser.add_argument('-exp', '--experiment', type=int, default=0, help='experiment number')
    parser.add_argument('-w', '--workers', type=int, default=1, help='number of workers for the fit function')
    # parser.add_argument('-s', '--steps', type=int, default=10000, help='steps per epoch')
    parser.add_argument('path')
    args = parser.parse_args()
    return args