import datetime
import os
import time

import torch.backends.cudnn as cudnn
import numpy as np

from utils.box_cars_dataset import BoxCarsDataset
from models.posenet import PoseNet
from torch import optim

# cudnn.benchmark = True
# cudnn.enabled = True

import torch
import argparse

from tqdm import tqdm


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', type=str, default=None, help='resume from file')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-n', '--num_stacks', type=int, default=2, help='number of stacks')
    parser.add_argument('-i', '--input_size', type=int, default=128, help='size of input')
    parser.add_argument('-o', '--heatmap_size', type=int, default=64, help='size of output heatmaps')
    parser.add_argument('-e', '--epochs', type=int, default=250, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('-bn', '--batch_norm', action='store_true', default=False)
    parser.add_argument('-exp', '--experiment', type=int, default=0, help='experiment number')
    # parser.add_argument('-s', '--steps', type=int, default=10000, help='steps per epoch')
    parser.add_argument('path')
    args = parser.parse_args()
    return args


def train():
    args = parse_command_line()
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

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

    checkpoint_dir_name = 'VP1VP2_{}in_{}out_{}s_{}n_{}_{}b_{}'.format(args.input_size, args.heatmap_size, len(scales), args.num_stacks, bn_str, args.batch_size, args.experiment)
    checkpoint_dir_path = os.path.join('checkpoints', checkpoint_dir_name)

    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)

    print("Checkpoint dir name: ", checkpoint_dir_name)

    model = PoseNet(args.num_stacks, args.input_size, 2 * len(scales), bn=args.batch_norm, init_reduction=args.input_size // args.heatmap_size).to(device)
    optimizer = optim.Adam(model.parameters())
    print("Model initialized")

    start_epoch = 0

    if args.resume is not None:
        print("Loading checkpoint {}".format(args.resume))

        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

        print("Starting from epoch: ", start_epoch)

    print("Loading dataset: ", args.path)
    train_data = BoxCarsDataset(args.path, 'train', img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    print("Training dataset loaded with {} samples".format(len(train_data)))

    val_data = BoxCarsDataset(args.path, 'val', img_size=args.input_size, heatmap_size=args.heatmap_size, scales=scales)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    print("Validation dataset loaded with {} samples".format(len(val_data)))

    for epoch in range(start_epoch, args.epochs):
        print('*' * 20)
        print("Starting training for epoch {}".format(epoch))
        epoch_start_time = time.time()
        losses = []
        accs = []
        for step, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = model.calc_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = model.calc_acc(outputs, labels).detach().cpu().numpy()

            losses.append(loss.item())
            accs.append(acc)

            remaining_time = (time.time() - epoch_start_time) / (step + 1) * (len(train_loader) - step)
            print("Epoch {}/{}, \t step {}/{}, \t loss: {}, \t mean acc: {}, \t remaining: {}".format(epoch, args.epochs, step, len(train_loader), loss, np.mean(acc), datetime.timedelta(seconds=remaining_time)))

        loss = np.mean(losses)
        acc = np.mean(np.array(accs), axis=0)

        print('*' * 20)
        print("Starting validation for epoch {}".format(epoch))

        with torch.no_grad():
            val_losses = []
            val_accs = []
            for step, data in enumerate(tqdm(val_loader)):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = model.calc_loss(outputs, labels)
                acc = model.calc_acc(outputs, labels).detach().cpu().numpy()

                val_losses.append(loss.item())
                val_accs.append(acc)

                # remaining_time = (time.time() - epoch_start_time) / (step + 1) * (len(val_loader) - step)
                # print("Epoch {}/{}, \t step {}/{}, \t val loss: {}, \t val mean acc: {}, \t remaining: {}"
                #       .format(epoch, args.epochs, step, len(val_loader), loss, np.mean(acc), datetime.timedelta(seconds=remaining_time)))

            val_loss = np.mean(losses)
            val_acc = np.mean(np.array(accs), axis=0)

            print("Epoch {}/{}, \t val loss: {}, \t val mean acc: {}, all accs:".format(epoch, args.epochs, val_loss, val_acc.mean()))
            print(val_acc)

        save_path = os.path.join(checkpoint_dir_path, '{:03d}.pth'.format(epoch))
        print('*' * 20)
        print("Saving model to ", save_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'mean_acc': np.mean(acc),
            'accs': acc,
            'val_loss': val_loss,
            'val_mean_acc': np.mean(val_acc),
            'val_accs': val_acc
        }, save_path)
        print("Model saved")



if __name__ == '__main__':
    train()
