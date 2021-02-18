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
        loss = keras.losses.mse


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
