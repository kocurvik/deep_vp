import argparse
import os

from tensorflow import keras

from models.hourglass import create_hourglass_network, heatmap_mean_accuracy
from utils.box_cars_dataset import BoxCarsDataset


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

def train():
    args = parse_command_line()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    scales = [0.03, 0.1, 0.3, 1.0]

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
    print("Workers: ", args.workers)
    print("Use multiprocessing: ", args.workers > 1)

    if args.mobilenet:
        module = 'mobilenet'
        module_str = 'm'
    else:
        module = 'bottleneck'
        module_str = 'b'


    snapshot_dir_name = 'VP1VP2{}_{}in_{}out_{}s_{}n_{}b_{}c_{}'.format(module_str, args.input_size, args.heatmap_size, len(scales), args.num_stacks, args.batch_size, args.channels, args.experiment)
    snapshot_dir_path = os.path.join('snapshots', snapshot_dir_name)

    if not os.path.exists(snapshot_dir_path):
        os.makedirs(snapshot_dir_path)

    print("Checkpoint dir name: ", snapshot_dir_name)

    model = create_hourglass_network(2 * len(scales), args.num_stacks, inres=args.input_size, outres=args.heatmap_size, bottleneck=module, num_channels=args.channels)

    if args.resume:
        model.load_weights(os.path.join(snapshot_dir_path, 'model.{:03d}.h5'.format(args.resume)))


    adam = keras.optimizers.Adam(args.lr)
    model.compile(adam, 'MSE', metrics=[heatmap_mean_accuracy(args.batch_size, args.heatmap_size, len(scales) * 2)])

    print(model.summary())

    train_dataset = BoxCarsDataset(args.path, 'train', batch_size=args.batch_size, img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales)
    val_dataset = BoxCarsDataset(args.path, 'val', batch_size=args.batch_size, img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales)

    callbacks = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(snapshot_dir_path, 'model.{epoch:03d}.h5')),
                 keras.callbacks.TensorBoard(log_dir=os.path.join('logs', snapshot_dir_name))]

    print("Starting training with lr: {}".format(args.lr))

    model.fit_generator(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=callbacks, initial_epoch=args.resume, workers=args.workers, use_multiprocessing=args.workers > 1)

    if args.shutdown:
        os.system('sudo poweroff')

if __name__ == '__main__':
    train()