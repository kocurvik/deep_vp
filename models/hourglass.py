import argparse
import os

from tensorflow import keras
import tensorflow as tf

# Adapted from: https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras/

def heatmap_mean_accuracy(batch_size, heatmap_size, scale_size):
    def mean_acc(y_pred, y_gt):
        argmax_pred = tf.argmax(tf.reshape(y_pred, [batch_size, heatmap_size * heatmap_size, scale_size]), axis=1)
        argmax_gt = tf.argmax(tf.reshape(y_gt, [batch_size, heatmap_size * heatmap_size, scale_size]), axis=1)
        eq = tf.cast(tf.equal(argmax_pred, argmax_gt), "float32")
        return keras.backend.mean(eq)

    return mean_acc

def create_hourglass_network(num_classes, num_stacks, inres=128, outres=128, num_channels=256, bottleneck='bottleneck'):
    input = keras.layers.Input(shape=(inres, inres, 3))
    init_reduction = inres // outres

    if bottleneck == 'mobilenet':
        bottleneck = bottleneck_mobile
    else:
        bottleneck = bottleneck_block

    front_features = create_front_module(input, num_channels, bottleneck, init_reduction)

    head_next_stage = front_features

    outputs = []
    for i in range(num_stacks):
        head_next_stage, head_to_loss = hourglass_module(head_next_stage, num_classes, num_channels, bottleneck, i)
        outputs.append(head_to_loss)

    model = keras.models.Model(inputs=input, outputs=outputs)
    # rms = RMSprop(lr=5e-4)
    # model.compile(optimizer=rms, loss=mean_squared_error, metrics=["accuracy"])
    return model


def hourglass_module(bottom, num_classes, num_channels, bottleneck, hgid):
    # create left features , f1, f2, f4, and f8
    left_features = create_left_half_blocks(bottom, bottleneck, hgid, num_channels)

    # create right features, connect with left features
    rf1 = create_right_half_blocks(left_features, bottleneck, hgid, num_channels)

    # add 1x1 conv with two heads, head_next_stage is sent to next stage
    # head_parts is used for intermediate supervision
    head_next_stage, head_parts = create_heads(bottom, rf1, num_classes, hgid, num_channels)

    return head_next_stage, head_parts


def bottleneck_block(bottom, num_out_channels, block_name):
    # skip layer
    if keras.backend.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = keras.layers.Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                       name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = keras.layers.Conv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x1')(bottom)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.Conv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                name=block_name + '_conv_3x3_x2')(_x)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.Conv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                name=block_name + '_conv_1x1_x3')(_x)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.Add(name=block_name + '_residual')([_skip, _x])

    return _x


def bottleneck_mobile(bottom, num_out_channels, block_name):
    # skip layer
    if keras.backend.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = keras.layers.SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                                name=block_name + 'skip')(bottom)

    # residual: 3 conv blocks,  [num_out_channels/2  -> num_out_channels/2 -> num_out_channels]
    _x = keras.layers.SeparableConv2D(num_out_channels // 2, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_x1')(bottom)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.SeparableConv2D(num_out_channels // 2, kernel_size=(3, 3), activation='relu', padding='same',
                         name=block_name + '_conv_3x3_x2')(_x)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.SeparableConv2D(num_out_channels, kernel_size=(1, 1), activation='relu', padding='same',
                         name=block_name + '_conv_1x1_x3')(_x)
    _x = keras.layers.BatchNormalization()(_x)
    _x = keras.layers.Add(name=block_name + '_residual')([_skip, _x])

    return _x


def create_front_module(input, num_channels, bottleneck, init_reduction=1):
    # front module, input to 1/4 resolution
    # 1 7x7 conv + maxpooling
    # 3 residual block

    if init_reduction == 4:
        _x = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', name='front_conv_1x1_x1')(input)
        _x = keras.layers.BatchNormalization()(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x1')
        _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x2')
        _x = bottleneck(_x, num_channels, 'front_residual_x3')

    elif init_reduction == 2:
        _x = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', name='front_conv_1x1_x1')(input)
        _x = keras.layers.BatchNormalization()(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x1')
        _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x2')
        _x = bottleneck(_x, num_channels, 'front_residual_x3')

    else:
        _x = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu', name='front_conv_1x1_x1')(input)
        _x = keras.layers.BatchNormalization()(_x)
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x1')
        _x = bottleneck(_x, num_channels // 2, 'front_residual_x2')
        _x = bottleneck(_x, num_channels, 'front_residual_x3')

    return _x


def create_left_half_blocks(bottom, bottleneck, hglayer, num_channels):
    # create left half blocks for hourglass module
    # f1, f2, f4 , f8 : 1, 1/2, 1/4 1/8 resolution

    hgname = 'hg' + str(hglayer)

    f1 = bottleneck(bottom, num_channels, hgname + '_l1')
    _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f1)

    f2 = bottleneck(_x, num_channels, hgname + '_l2')
    _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f2)

    f4 = bottleneck(_x, num_channels, hgname + '_l4')
    _x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(f4)

    f8 = bottleneck(_x, num_channels, hgname + '_l8')

    return (f1, f2, f4, f8)


def connect_left_to_right(left, right, bottleneck, name, num_channels):
    '''
    :param left: connect left feature to right feature
    :param name: layer name
    :return:
    '''
    # left -> 1 bottlenect
    # right -> upsampling
    # keras.layers.Add   -> left + right

    _xleft = bottleneck(left, num_channels, name + '_connect')
    _xright = keras.layers.UpSampling2D()(right)
    add = keras.layers.Add()([_xleft, _xright])
    out = bottleneck(add, num_channels, name + '_connect_conv')
    return out


def bottom_layer(lf8, bottleneck, hgid, num_channels):
    # blocks in lowest resolution
    # 3 bottlenect blocks + keras.layers.Add

    lf8_connect = bottleneck(lf8, num_channels, str(hgid) + "_lf8")

    _x = bottleneck(lf8, num_channels, str(hgid) + "_lf8_x1")
    _x = bottleneck(_x, num_channels, str(hgid) + "_lf8_x2")
    _x = bottleneck(_x, num_channels, str(hgid) + "_lf8_x3")

    rf8 = keras.layers.Add()([_x, lf8_connect])

    return rf8


def create_right_half_blocks(leftfeatures, bottleneck, hglayer, num_channels):
    lf1, lf2, lf4, lf8 = leftfeatures

    rf8 = bottom_layer(lf8, bottleneck, hglayer, num_channels)

    rf4 = connect_left_to_right(lf4, rf8, bottleneck, 'hg' + str(hglayer) + '_rf4', num_channels)

    rf2 = connect_left_to_right(lf2, rf4, bottleneck, 'hg' + str(hglayer) + '_rf2', num_channels)

    rf1 = connect_left_to_right(lf1, rf2, bottleneck, 'hg' + str(hglayer) + '_rf1', num_channels)

    return rf1


def create_heads(prelayerfeatures, rf1, num_classes, hgid, num_channels):
    # two head, one head to next stage, one head to intermediate features
    head = keras.layers.Conv2D(num_channels, kernel_size=(1, 1), activation='relu', padding='same', name=str(hgid) + '_conv_1x1_x1')(
        rf1)
    head = keras.layers.BatchNormalization()(head)

    # for head as intermediate supervision, use 'linear' as activation.
    head_parts = keras.layers.Conv2D(num_classes, kernel_size=(1, 1), activation='linear', padding='same',
                        name=str(hgid) + '_out')(head)

    # use linear activation
    head = keras.layers.Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                  name=str(hgid) + '_conv_1x1_x2')(head)
    head_m = keras.layers.Conv2D(num_channels, kernel_size=(1, 1), activation='linear', padding='same',
                    name=str(hgid) + '_conv_1x1_x3')(head_parts)

    head_next_stage = keras.layers.Add()([head, head_m, prelayerfeatures])
    return head_next_stage, head_parts


def load_model(args, scales):
    print("Initializing model")
    print("Batch size: ", args.batch_size)
    print("Num stacks: ", args.num_stacks)
    print("Input size: {} x {}".format(args.input_size, args.input_size))
    print("Heatmap size: {} x {}".format(args.heatmap_size, args.heatmap_size))
    print("Training for {} epochs".format(args.epochs))
    print("Scales: ", scales)
    print("Channels: ", args.channels)
    print("Experiment number: ", args.experiment)
    print("Mobilenet version: ", args.mobilenet)
    print("Heatmap distribution constructed in original coords: ", args.peak_original)

    if args.mobilenet:
        module = 'mobilenet'
        module_str = 'm'
    else:
        module = 'bottleneck'
        module_str = 'b'

    peak_str = 'po' if args.peak_original else 'pd'

    snapshot_dir_name = 'VP1VP2{}_{}_{}in_{}out_{}s_{}n_{}b_{}c_{}'.format(module_str, peak_str, args.input_size, args.heatmap_size,
                                                                        len(scales), args.num_stacks, args.batch_size,
                                                                        args.channels, args.experiment)
    snapshot_dir_path = os.path.join('snapshots', snapshot_dir_name)

    if not os.path.exists(snapshot_dir_path):
        os.makedirs(snapshot_dir_path)

    print("Checkpoint dir name: ", snapshot_dir_name)

    model = create_hourglass_network(2 * len(scales), args.num_stacks, inres=args.input_size, outres=args.heatmap_size,
                                     bottleneck=module, num_channels=args.channels)

    if args.resume:



        if args.experiment_resume is None:
            args.experiment_resume = args.experiment

        snapshot_dir_name = 'VP1VP2{}_{}_{}in_{}out_{}s_{}n_{}b_{}c_{}'.format(module_str, peak_str, args.input_size,
                                                                               args.heatmap_size,
                                                                               len(scales), args.num_stacks,
                                                                               args.batch_size,
                                                                               args.channels, args.experiment_resume)

        resume_model_path = os.path.join(resume_dir_name, 'model.{:03d}.h5'.format(args.resume))
        print("Loading model", resume_model_path)
        model.load_weights(resume_model_path)

    return model, snapshot_dir_name, snapshot_dir_path


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
    parser.add_argument('-po', '--peak_original', action='store_true', default=False, help='whether to construct the peak in the original space')
    parser.add_argument('--shutdown', action='store_true', default=False, help='shutdown the machine when done')
    parser.add_argument('-c', '--channels', type=int, default=256, help='number of channels in network')
    parser.add_argument('-exp', '--experiment', type=int, default=0, help='experiment number')
    parser.add_argument('-expr', '--experiment-resume', type=int, default=None, help='experiment number to resume from')
    parser.add_argument('-w', '--workers', type=int, default=1, help='number of workers for the fit function')
    # parser.add_argument('-s', '--steps', type=int, default=10000, help='steps per epoch')
    parser.add_argument('path')
    args = parser.parse_args()
    return args