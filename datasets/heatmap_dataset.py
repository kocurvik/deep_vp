import os
import pickle

import cv2
import numpy as np

from tensorflow import keras
from utils.diamond_space import diamond_coords_from_original, vp_to_heatmap, heatmap_to_orig, process_heatmap_old


class GenerateHeatmap():
    def __init__(self, output_res, scales):
        self.output_res = output_res
        self.scales = scales
        self.sigma = self.output_res/64
        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

    def __call__(self, vps):
        hms = np.zeros(shape = (self.output_res, self.output_res, len(vps) * len(self.scales)), dtype = np.float32)
        for vp_idx, vp in enumerate(vps):
            for scale_idx, scale in enumerate(self.scales):
                idx = len(self.scales) * vp_idx + scale_idx

                vp_heatmap = vp_to_heatmap(vp, self.output_res, scale=scale)

                # vp_heatmap = (vp_diamond + 0.5) * self.output_res
                # vp_heatmap = ((self.R @ vp_diamond.T)) * (np.sqrt(2) / 2 * self.output_res) + self.output_res / 2
                # self.R = np.array([[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]])

                x, y = int(vp_heatmap[0]), int(vp_heatmap[1])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue

                ul = int(y - 3*self.sigma - 1), int(x - 3*self.sigma - 1)
                br = int(y + 3*self.sigma + 2), int(x + 3*self.sigma + 2)
                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]
                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)

                hms[aa:bb, cc:dd, idx] = np.maximum(hms[aa:bb, cc:dd, idx], self.g[a:b, c:d])
        return hms


class HeatmapBoxCarsDataset(keras.utils.Sequence):
    def __init__(self, path, split, batch_size=32, img_size=128, heatmap_size=128, scales=(0.1, 0.3, 1.0, 3, 10.0), peak_original=False, perspective_sigma=25.0, crop_delta=10):
        'Initialization'
        with open(os.path.join(path, 'dataset.pkl'), 'rb') as f:
            self.data = pickle.load(f, encoding="latin-1", fix_imports=True)

        with open(os.path.join(path, 'atlas.pkl'), 'rb') as f:
            self.atlas = pickle.load(f, encoding="latin-1", fix_imports=True)

        self.split = split
        self.img_dir = os.path.join(path, 'images')

        self.batch_size = batch_size

        self.img_size = img_size
        self.heatmap_size = heatmap_size

        self.scales = scales

        if peak_original:
            self.orig_coord_heatmaps = []
            for scale in scales:
                orig_coord_heatmap = heatmap_to_orig(heatmap_size, scale=scale)
                # make nans inf to calc inf distance
                orig_coord_heatmap[np.isnan(orig_coord_heatmap)] = np.inf
                self.orig_coord_heatmaps.append(orig_coord_heatmap)
        else:
            self.generate_heatmaps = GenerateHeatmap(self.heatmap_size, self.scales)

        self.perspective_sigma = perspective_sigma
        self.crop_delta = crop_delta
        self.aug = perspective_sigma > 0 or crop_delta > 0

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
        heatmaps = []
        for i in actual_idxs:
            img, heatmap = self.get_single_item(i)
            imgs.append(img)
            heatmaps.append(heatmap)

        return np.array(imgs), [np.array(heatmaps), np.array(heatmaps)]

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
        rect_dst = 2.0 * rect_dst.astype(np.float32) #+ self.perspective_sigma * 4

        if np.random.rand() > 0.5:
            rect_src = np.array([[img.shape[1], 0], [0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]],
                                dtype=np.float32)

        rect_dst[:, 0] -= np.min(rect_dst[:, 0]) - self.crop_delta
        rect_dst[:, 1] -= np.min(rect_dst[:, 1]) - self.crop_delta

        M = cv2.getPerspectiveTransform(rect_src[:, :], rect_dst[:, :])

        bbox_warped = cv2.perspectiveTransform(bbox[:, np.newaxis, :], M)

        max_x = min(int(np.max(bbox_warped[:, 0, 0])) + self.crop_delta, 900)
        max_y = min(int(np.max(bbox_warped[:, 0, 1])) + self.crop_delta, 900)

        img_warped = cv2.warpPerspective(img, M, (max_x, max_y), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        vp1_warped = cv2.perspectiveTransform(vp1[np.newaxis, np.newaxis, :], M)
        vp2_warped = cv2.perspectiveTransform(vp2[np.newaxis, np.newaxis, :], M)
        # cv2.imshow("Warped", img_warped)

        return img_warped, bbox_warped[:, 0, :], vp1_warped[0, 0], vp2_warped[0, 0]

    def generate_heatmaps(self, vps):
        heatmaps = np.empty([self.heatmap_size, self.heatmap_size, len(vps) * len(self.orig_coord_heatmaps)])
        for i, vp in enumerate(vps):
            for j, orig_coord_heatmap in enumerate(self.orig_coord_heatmaps):
                # exp(- 1/2 * (d/sigma)**2)
                d_sqr = np.sum((orig_coord_heatmap - vp[np.newaxis, np.newaxis, :]) ** 2, axis=-1)

                vp_heatmap_int = np.round(vp_to_heatmap(vp, self.heatmap_size, scale=self.scales[j])).astype(np.int)
                i_min, i_max = max(vp_heatmap_int[0] - 3, 0), min(vp_heatmap_int[0] + 4, self.heatmap_size)
                j_min, j_max = max(vp_heatmap_int[1] - 3, 0), min(vp_heatmap_int[1] + 4, self.heatmap_size)

                sigma_dist_sqr = np.nanpercentile(d_sqr[i_min: i_max, j_min: j_max], 20)
                neg_half_times_inv_sigma_sqr = 1.0 / sigma_dist_sqr
                h = np.exp(- d_sqr * neg_half_times_inv_sigma_sqr)

                heatmaps[:, :, i * len(self.orig_coord_heatmaps) + j] = h / (np.sum(h) + 1e-8)

        return heatmaps

    def generate_item(self, img, bbox, vp1, vp2):
        tries = 0

        while True:

            if tries < 4 and self.split == 'train' and self.aug:
                warped_img, warped_bbox, warped_vp1, warped_vp2 = self.random_perspective_transform(img, bbox, vp1, vp2)
                x_min = int(max(np.floor(np.min(warped_bbox[:, 0])) + np.random.randint(-self.crop_delta, self.crop_delta), 0))
                x_max = int(min(np.ceil(np.max(warped_bbox[:, 0])) + np.random.randint(-self.crop_delta, self.crop_delta), warped_img.shape[1]))
                y_min = int(max(np.floor(np.min(warped_bbox[:, 1])) + np.random.randint(-self.crop_delta, self.crop_delta), 0))
                y_max = int(min(np.ceil(np.max(warped_bbox[:, 1])) + np.random.randint(-self.crop_delta, self.crop_delta), warped_img.shape[0]))

            else:
                # warped_img, warped_bbox, warped_vp1, warped_vp2 = self.random_perspective_transform(img, bbox, vp1, vp2, force_no_perspective=True)
                warped_img, warped_bbox, warped_vp1, warped_vp2 = img, bbox, vp1, vp2
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


        # print("vp1: ", warped_vp1)
        # print("vp2: ", warped_vp2)
        #
        # print("vp1 diamond:",  diamond_coords_from_original(warped_vp1, 1.0))
        # print("vp2 diamond:",  diamond_coords_from_original(warped_vp2, 1.0))

        heatmaps = self.generate_heatmaps([warped_vp1, warped_vp2])

        out_img = warped_img / 255
        out_heatmaps = heatmaps
        # out_img = transforms.ToTensor()(out_img)
        # out_img = torch.from_numpy(out_img).float()
        # out_heatmap = torch.from_numpy(heatmap).float()

        return out_img, out_heatmaps


def get_mean_heatmap_vp(heatmap, orig_coord_heatmap):
    vp_x_avg = np.average(orig_coord_heatmap[:, :, 0], weights=heatmap)
    vp_y_avg = np.average(orig_coord_heatmap[:, :, 1], weights=heatmap)
    vp_x_std = np.sqrt(np.average((orig_coord_heatmap[:, :, 0] - vp_x_avg) ** 2, weights=heatmap))
    vp_y_std = np.sqrt(np.average((orig_coord_heatmap[:, :, 1] - vp_y_avg) ** 2, weights=heatmap))

    return np.array([vp_x_avg, vp_y_avg]), np.array([vp_x_std, vp_y_std])

if __name__ == '__main__':
    path = 'D:/Skola/PhD/Data/BoxCars116k/'

    scales = [0.03, 0.1, 0.3, 1.0]
    heatmap_out = 64
    # scales = [0.1, 1.0]
    peak_original = False

    orig_coord_heatmaps = []
    for scale in scales:
        orig_coord_heatmap = heatmap_to_orig(heatmap_out, scale=scale)
        # make nans inf to calc inf distance
        orig_coord_heatmap[np.isnan(orig_coord_heatmap)] = 0
        orig_coord_heatmap[np.isinf(orig_coord_heatmap)] = 0
        orig_coord_heatmaps.append(orig_coord_heatmap)

    # d_aug = HeatmapBoxCarsDataset(path, 'train', img_size=128, heatmap_size=heatmap_out, scales=scales, peak_original=peak_original)
    # d_noaug = HeatmapBoxCarsDataset(path, 'train', img_size=128, heatmap_size=heatmap_out, scales=scales, peak_original=peak_original, perspective_sigma=0.0, crop_delta=0)
    d = HeatmapBoxCarsDataset(path, 'val', img_size=128, heatmap_size=heatmap_out, scales=scales, peak_original=peak_original, perspective_sigma=0.0, crop_delta=0)
    # print("Dataset loaded with size: ", len(d.instance_list))

    # cum_heatmap = np.zeros([heatmap_out, heatmap_out, 2*len(scales)])
    cum_heatmap_aug = np.zeros([heatmap_out, heatmap_out, 2*len(scales)])
    cum_heatmap_noaug = np.zeros([heatmap_out, heatmap_out, 2*len(scales)])

    # for _ in range(10000):
    #     i = np.random.choice(len(d_aug.instance_list))
    for i in [1856, 3815, 3611]:
        print("idx: ", i)

        img, heatmap = d.get_single_item(i)

        cv2.imwrite("vis/img_{}.png".format(i), img)
        # stop = [False, False]

        for vp_idx in range(2):
            for scale_idx, scale in enumerate(scales):
                idx = len(scales) * vp_idx + scale_idx
                cv2.imwrite("vis/heatmap_{}_vp{}_s{}.png".format(i, vp_idx + 1, scale), cv2.applyColorMap(np.uint8(255 * heatmap[:, :, idx] / np.max(heatmap[:, :, idx])), cv2.COLORMAP_PARULA))

        # img, heatmap_aug = d_aug.get_single_item(i)
        # img, heatmap_noaug = d_noaug.get_single_item(i)

        # cum_heatmap_aug += heatmap_aug
        # cum_heatmap_noaug += heatmap_noaug

    # for vp_idx in range(2):
        # for scale_idx, scale in enumerate(scales):
        #     idx = len(scales) * vp_idx + scale_idx
        #     cv2.imwrite("vis/heatmap_aug_vp{}_scale{}.png".format(vp_idx + 1, scale), cv2.applyColorMap(np.uint8(255 * cum_heatmap_aug[:, :, idx] / np.max(cum_heatmap_aug[:, :, idx])), cv2.COLORMAP_PARULA))
        #     cv2.imwrite("vis/heatmap_noaug_vp{}_scale{}.png".format(vp_idx + 1, scale), cv2.applyColorMap(np.uint8(255 * cum_heatmap_noaug[:, :, idx] / np.max(cum_heatmap_noaug[:, :, idx])), cv2.COLORMAP_PARULA))




        # cv2.imshow("Img", img)
        # stop = [False, False]
        #
        # for vp_idx in range(2):
        #     for scale_idx, scale in enumerate(scales):
        #         idx = len(scales) * vp_idx + scale_idx
        #         cv2.imshow("Cummulative heatmap for vp{} at scale {}".format(vp_idx + 1, scale), cum_heatmap[:, :, idx] / np.max(cum_heatmap[:, :, idx]))
        #         cv2.imshow("Heatmap for vp{} at scale {}".format(vp_idx + 1, scale), heatmap[:, :, idx]/np.max(heatmap[:, :, idx]))
        #         if peak_original:
        #             vp, std = get_mean_heatmap_vp(heatmap[:, :, idx], orig_coord_heatmaps[scale_idx])
        #         else:
        #             vp, std  = process_heatmap_old(heatmap[:, :, idx], scale)
        #
        #         print(vp, std)

            # if np.abs(vp[0]) < 7 and np.abs(vp[1]) < 3:
            #     stop[vp_idx] = True

        #
        # if stop[0] and stop[1]:
        #     cv2.imwrite("img_{}.png".format(i), 255 * img)
        #     cv2.imwrite("heatmap_vp1_{}.png".format(i), 255 * heatmap[:, :, 3])
        #     cv2.imwrite("heatmap_vp2_{}.png".format(i), 255 * heatmap[:, :, -1])
        #
        #     cv2.imwrite("heatmap_cmap_vp1_{}.png".format(i), cv2.applyColorMap(np.uint8(255 * heatmap[:, :, 3]), cv2.COLORMAP_PARULA))
        #     cv2.imwrite("heatmap_cmap_vp2_{}.png".format(i), cv2.applyColorMap(np.uint8(255 * heatmap[:, :, -1]), cv2.COLORMAP_PARULA))
        #
        #     cv2.waitKey(1)
        # else:
        #     cv2.waitKey(1)




