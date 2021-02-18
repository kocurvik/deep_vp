import os

from models.reg import load_model, vp1_dist, vp2_dist, parse_command_line
from tensorflow import keras
from utils.gpu import set_gpus
from utils.reg_dataset import RegBoxCarsDataset


def train():
    args = parse_command_line()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_gpus()

    model, loss, snapshot_dir_name, snapshot_dir_path = load_model(args)

    print("Loading dataset!")
    train_dataset = RegBoxCarsDataset(args.path, 'train', batch_size=args.batch_size, img_size=args.input_size)
    print("Loaded training dataset with {} samples".format(len(train_dataset.instance_list)))
    val_dataset = RegBoxCarsDataset(args.path, 'val', batch_size=args.batch_size, img_size=args.input_size)
    print("Loaded val dataset with {} samples".format(len(val_dataset.instance_list)))

    # callbacks = [keras.callbacks.ModelCheckpoint(filepath=os.path.join(snapshot_dir_path, 'model.{epoch:03d}.h5')),
    #              keras.callbacks.TensorBoard(log_dir=os.path.join('logs', snapshot_dir_name))]
    callbacks = []

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