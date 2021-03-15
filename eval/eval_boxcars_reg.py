import os

import numpy as np
from models.reg import parse_command_line, load_model
from utils.gpu import set_gpus
from datasets.reg_dataset import RegBoxCarsDataset

def eval():
    args = parse_command_line()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    set_gpus()

    scales = [0.03, 0.1, 0.3, 1.0]

    model, _, snapshot_dir_name, _ = load_model(args)
    print("Reg model loaded")

    test_dataset = RegBoxCarsDataset(args.path, 'test', batch_size=args.batch_size_eval, img_size=args.input_size, num_stacks=1,
                                     use_diamond=False, scale=1.0, perspective_sigma=0.0, crop_delta=0)

    gt_vp_list = []
    pred_vp_list = []
    pred_dists_vars = []

    for X, gt_vp in test_dataset:
        pred_vps = model.predict(X)

        gt_vp_list.append(gt_vp[0])
        pred_vp_list.append(pred_vps)

    gt_vps = np.concatenate(gt_vp_list, axis=0)
    pred_vps = np.concatenate(pred_vp_list, axis=0) / args.scale

    diff = pred_vps - gt_vps
    diff[np.isinf(diff)] = np.nan
    vp1_d = np.linalg.norm(diff[:, :2], axis=-1)
    vp2_d = np.linalg.norm(diff[:, 2:], axis=-1)

    vp1_gt_norm = np.linalg.norm(gt_vps[:, :2], axis=-1)
    vp2_gt_norm = np.linalg.norm(gt_vps[:, 2:], axis=-1)

    print('*' * 80)
    print("Median vp1 abs distance: {}".format(np.nanmedian(vp1_d)))
    print("Median vp2 abs distance: {}".format(np.nanmedian(vp2_d)))
    print("Mean vp1 abs distance: {}".format(np.nanmean(vp1_d)))
    print("Mean vp2 abs distance: {}".format(np.nanmean(vp2_d)))
    print("Median vp1 rel distance: {}".format(np.nanmedian(vp1_d / vp1_gt_norm)))
    print("Median vp2 rel distance: {}".format(np.nanmedian(vp2_d / vp2_gt_norm)))
    print("Mean vp1 rel distance: {}".format(np.nanmean(vp1_d / vp1_gt_norm)))
    print("Mean vp2 rel distance: {}".format(np.nanmean(vp2_d / vp2_gt_norm)))

if __name__ == '__main__':
    eval()