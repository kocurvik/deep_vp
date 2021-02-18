import argparse
import os

from models.reg import load_model, vp1_dist, vp2_dist
from tensorflow import keras
from utils.gpu import set_gpus
from utils.reg_dataset import RegBoxCarsDataset

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
    parser.add_argument('-exp', '--experiment', type=int, default=0, help='experiment number')
    parser.add_argument('-w', '--workers', type=int, default=1, help='number of workers for the fit function')
    parser.add_argument('path')
    args = parser.parse_args()
    return args

def train():
    args = parse_command_line()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_gpus()

    model, loss, snapshot_dir_name, snapshot_dir_path = load_model(args)

    print("Loading dataset!")
    train_dataset = RegBoxCarsDataset(args.path, 'train', batch_size=args.batch_size, img_size=args.input_size)
    print("Loaded training dataset with {} samples".format(len(train_dataset)))
    val_dataset = RegBoxCarsDataset(args.path, 'val', batch_size=args.batch_size, img_size=args.input_size)
    print("Loaded val dataset with {} samples".format(len(val_dataset)))

    callbacks = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(snapshot_dir_path, 'model.{epoch:03d}.h5')),
                 keras.callbacks.TensorBoard(log_dir=os.path.join('logs', snapshot_dir_name))]

    print("Workers: ", args.workers)
    print("Use multiprocessing: ", args.workers > 1)
    print("Starting training with lr: {}".format(args.lr))

    adam = keras.optimizers.Adam(args.lr)
    model.compile(adam, loss, metrics=[vp1_dist, vp2_dist])

    model.fit_generator(train_dataset, validation_data=val_dataset, epochs=args.epochs, callbacks=callbacks,
                        initial_epoch=args.resume, workers=args.workers, use_multiprocessing=args.workers > 1)

    if args.shutdown:
        os.system('sudo poweroff')


if __name__ == '__main__':
    train()