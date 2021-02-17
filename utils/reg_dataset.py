import os
import pickle

import cv2
import numpy as np
from tensorflow import keras


class RegBoxCarsDataset(keras.utils.Sequence):
    def __init__(self, path, split, batch_size=32, img_size=128, perspective_sigma=25.0, crop_delta=10):
        'Initialization'
        with open(os.path.join(path, 'dataset.pkl'), 'rb') as f:
            self.data = pickle.load(f, encoding="latin-1", fix_imports=True)

        with open(os.path.join(path, 'atlas.pkl'), 'rb') as f:
            self.atlas = pickle.load(f, encoding="latin-1", fix_imports=True)

        self.split = split
        self.img_dir = os.path.join(path, 'images')

        self.batch_size = batch_size

        self.img_size = img_size

        self.perspective_sigma = perspective_sigma
        self.crop_delta = crop_delta

        self.instance_list = []

        # generate split every tenth sample is validation - remove useless samples from atlas
        for s_idx, sample in enumerate(self.data['samples']):
            if s_idx % 10 == 0:
                if self.split != 'val':
                    self.atlas[s_idx] = None
                else:
                    for i_idx, instance in enumerate(sample['instances']):
                        self.instance_list.append((s_idx, i_idx))
            elif s_idx % 10 == 1:
                if self.split != 'test':
                    self.atlas[s_idx] = None
                else:
                    for i_idx, instance in enumerate(sample['instances']):
                        self.instance_list.append((s_idx, i_idx))
            else:
                if self.split != 'train':
                    self.atlas[s_idx] = None
                else:
                    for i_idx, instance in enumerate(sample['instances']):
                        self.instance_list.append((s_idx, i_idx))

        self.idxs = np.arange(len(self.instance_list))
        if self.split == 'train':
            np.random.shuffle(self.idxs)

    def __len__(self):
        'Denotes the total number of samples'
        # return len(self.instance_list)
        return int(np.floor(len(self.instance_list) / self.batch_size))
        # return 100

    def __getitem__(self, idx):
        actual_idxs = self.idxs[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs = []
        vps = []
        for i in actual_idxs:
            img, vp = self.get_single_item(i)
            imgs.append(img)
            vps.append(vp)

        return np.array(imgs), np.array(vps)

    def on_epoch_end(self):
        if self.split == 'train':
            np.random.shuffle(self.idxs)

    def get_single_item(self, index):
        s_idx, i_idx = self.instance_list[index]

        sample = self.data['samples'][s_idx]
        instance = sample['instances'][i_idx]
        camera = self.data['cameras'][sample['camera']]

        bbox = instance['3DBB'] - instance['3DBB_offset']

        vp1 = camera['vp1'] - instance['3DBB_offset']
        vp2 = camera['vp2'] - instance['3DBB_offset']
        # vp3 = camera['vp3'] - instance['3DBB_offset']

        img = cv2.imdecode(self.atlas[s_idx][i_idx], 1)

        return self.generate_item(img, bbox, vp1, vp2)

    def random_perspective_transform(self, img, bbox, vp1, vp2, force_no_perspective=False):

        rect_src = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]], dtype=np.float32)
        if force_no_perspective:
            rect_dst = rect_src
        else:
            rect_dst = rect_src + self.perspective_sigma * np.random.randn(*rect_src.shape)
        rect_dst = 2 * rect_dst.astype(np.float32) + 100

        if np.random.rand() > 0.5:
            rect_src = np.array([[img.shape[1], 0], [0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]],
                                dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect_src[:, :], rect_dst[:, :])

        img_warped = cv2.warpPerspective(img, M, (900, 900), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        bbox_warped = cv2.perspectiveTransform(bbox[:, np.newaxis, :], M)
        vp1_warped = cv2.perspectiveTransform(vp1[np.newaxis, np.newaxis, :], M)
        vp2_warped = cv2.perspectiveTransform(vp2[np.newaxis, np.newaxis, :], M)
        # cv2.imshow("Warped", img_warped)

        return img_warped, bbox_warped[:, 0, :], vp1_warped[0, 0], vp2_warped[0, 0]

    def generate_item(self, img, bbox, vp1, vp2):
        tries = 0

        while True:

            if tries < 4 and self.split == 'train':
                warped_img, warped_bbox, warped_vp1, warped_vp2 = self.random_perspective_transform(img, bbox, vp1, vp2)
                x_min = int(max(np.floor(np.min(warped_bbox[:, 0])) + np.random.randint(-self.crop_delta, self.crop_delta), 0))
                x_max = int(min(np.ceil(np.max(warped_bbox[:, 0])) + np.random.randint(-self.crop_delta, self.crop_delta), warped_img.shape[1]))
                y_min = int(max(np.floor(np.min(warped_bbox[:, 1])) + np.random.randint(-self.crop_delta, self.crop_delta), 0))
                y_max = int(min(np.ceil(np.max(warped_bbox[:, 1])) + np.random.randint(-self.crop_delta, self.crop_delta), warped_img.shape[0]))

            else:
                warped_img, warped_bbox, warped_vp1, warped_vp2 = self.random_perspective_transform(img, bbox, vp1, vp2, force_no_perspective=True)
                x_min = int(max(np.floor(np.min(warped_bbox[:, 0])), 0))
                x_max = int(min(np.ceil(np.max(warped_bbox[:, 0])), warped_img.shape[1]))
                y_min = int(max(np.floor(np.min(warped_bbox[:, 1])), 0))
                y_max = int(min(np.ceil(np.max(warped_bbox[:, 1])), warped_img.shape[0]))
                warped_img = warped_img[max(y_min, 0): y_max, max(x_min, 0): x_max, :]
                break

            if y_min + 25 < min(y_max, warped_img.shape[0]) and x_min + 25 < min(x_max, warped_img.shape[1]):
                warped_img = warped_img[max(y_min, 0): y_max, max(x_min, 0): x_max, :]
                break

            tries += 1

        warped_img = cv2.resize(warped_img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        # convert vp to new coordinate system in which img top left is (-1, -1) and bottom right is (1, 1)
        warped_vp1 -= np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
        warped_vp2 -= np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
        warped_vp1[0] /= (x_max - x_min) / 2.0
        warped_vp2[0] /= (x_max - x_min) / 2.0
        warped_vp1[1] /= (y_max - y_min) / 2.0
        warped_vp2[1] /= (y_max - y_min) / 2.0

        out_img = warped_img / 255

        return out_img, np.concatenate([warped_vp1, warped_vp2])
