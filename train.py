import argparse
import os

from models.hourglass import create_hourglass_network, heatmap_mean_accuracy
from utils.box_cars_dataset import BoxCarsDataset


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', type=str, default=None, help='resume from file')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('-n', '--num_stacks', type=int, default=2, help='number of stacks')
    parser.add_argument('-i', '--input_size', type=int, default=128, help='size of input')
    parser.add_argument('-o', '--heatmap_size', type=int, default=64, help='size of output heatmaps')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('-bn', '--batch_norm', action='store_true', default=False)
    parser.add_argument('-exp', '--experiment', type=int, default=0, help='experiment number')
    # parser.add_argument('-s', '--steps', type=int, default=10000, help='steps per epoch')
    parser.add_argument('path')
    args = parser.parse_args()
    return args

def train():
    args = parse_command_line()

    scales = [0.03, 0.1, 0.3, 1.0]

    print("Initializing model")
    print("Batch size: ", args.batch_size)
    print("Num stacks: ", args.num_stacks)
    print("Input size: {} x {}".format(args.input_size, args.input_size))
    print("Heatmap size: {} x {}".format(args.heatmap_size, args.heatmap_size))
    print("Training for {} epochs".format(args.epochs))
    print("Using BatchNorm: {}".format(args.batch_norm))
    print("Scales: ", scales)

    if args.batch_norm:
        bn_str = 'BN'
    else:
        bn_str = 'noBN'

    snapshot_dir_name = 'VP1VP2_{}in_{}out_{}s_{}n_{}_{}b_{}'.format(args.input_size, args.heatmap_size, len(scales), args.num_stacks, bn_str, args.batch_size, args.experiment)
    snapshot_dir_path = os.path.join('snapshots', snapshot_dir_name)

    if not os.path.exists(snapshot_dir_path):
        os.makedirs(snapshot_dir_path)

    print("Checkpoint dir name: ", snapshot_dir_name)

    model = create_hourglass_network(2 * len(scales), args.num_stacks, inres=args.input_size, outres=args.heatmap_size, num_channels=256)
    model.compile('adam', 'mse', metrics=[heatmap_mean_accuracy(args.batch_size, args.heatmap_size, len(scales) * 2)])

    print(model.summary())

    train_dataset = BoxCarsDataset(args.path, 'train', batch_size=args.batch_size, img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales)
    val_dataset = BoxCarsDataset(args.path, 'val', batch_size=args.batch_size, img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales)

    model.fit_generator(train_dataset, validation_data=val_dataset, epochs=args.epochs)


if __name__ == '__main__':
    train()