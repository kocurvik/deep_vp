import os

import torch
from matplotlib import pyplot as plt

def parse_command_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def summary():
    args = parse_command_line()

    epochs = []
    losses = []
    val_losses = []
    mean_accs = []
    val_mean_accs = []

    for filename in sorted(os.listdir(args.path)):
        if '.pth' not in filename:
            continue

        checkpoint = torch.load(os.path.join(args.path, filename))

        epochs.append(checkpoint['epoch'])
        losses.append(checkpoint['loss'])
        val_losses.append(checkpoint['val_loss'])
        mean_accs.append(checkpoint['mean_acc'])
        val_mean_accs.append(checkpoint['val_mean_acc'])

    plt.plot(epochs, losses, label='train_loss')
    plt.plot(epochs, val_losses, label='val_loss')
    plt.legend(loc='best')
    plt.show()

    plt.plot(epochs, mean_accs, label='train_mean_acc')
    plt.plot(epochs, val_mean_accs, label='val_mean_acc')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    summary()