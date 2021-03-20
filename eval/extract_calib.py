import argparse
import json
import os
import pickle

import numpy as np
import cv2
from eval.eval_calib import get_projector, eval_pure_calibration
from eval.extract_vp_utils import save
from matplotlib import pyplot as plt
from utils.diamond_space import get_focal


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-c', '--conf', type=float, default=0.1)
    parser.add_argument('bcs_path')
    parser.add_argument('bcp_path')
    parser.add_argument('json_names', metavar='N', nargs='+')

    args = parser.parse_args()
    return args


def get_calib_dict(vp1, vp2, pp, scale=1.0):
    return {'cars': [], 'camera_calibration': {"pp": [pp[0], pp[1]], "scale": scale, "vp1": [vp1[0], vp1[1]], "vp2": [vp2[0], vp2[1]]}}


def show_vps(vp1s, vp2s, session):
    canvas = np.zeros([1080, 1920, 3])
    zero = np.array([1920 // 2 - 1920 // 40, 1080 // 2 - 1080 // 40])
    canvas = cv2.rectangle(canvas, (1920 // 2 - 1920 // 40, 1080 // 2 - 1080 // 40), (1920 // 2 + 1920 // 40, 1080 // 2 + 1080 // 40), (255, 255, 255))

    # compare_calib_json_path = os.path.join(data_path, 'results', session, 'system_SochorCVIU_ManualCalib_ManualScale.json')
    # with open(compare_calib_json_path, 'r') as f:
    #     compare_calib = json.load(f)
    # compare_vp1 = compare_calib['camera_calibration']['vp1']
    # compare_vp2 = compare_calib['camera_calibration']['vp2']
    # compare_vp1 = np.array(compare_vp1) / 10 + zero
    # compare_vp2 = np.array(compare_vp2) / 10 + zero
    #
    # cv2.circle(canvas, (int(compare_vp1[0]), int(compare_vp1[1])), radius=3, color=(0, 255, 0))
    # cv2.circle(canvas, (int(compare_vp2[0]), int(compare_vp2[1])), radius=3, color=(0, 0, 255))
    # cv2.line(canvas, (int(compare_vp1[0]), int(compare_vp1[1])), (int(compare_vp2[0]), int(compare_vp2[1])), color=(255, 0, 0), thickness=3)

    vp1s = vp1s / 40 + zero
    vp2s = vp2s / 40 + zero

    for vp1, vp2 in zip(vp1s[:100], vp2s[:100]):
        try:
            cv2.circle(canvas, (int(vp1[0]), int(vp1[1])), radius=1, color=(0, 255, 0))
            cv2.circle(canvas, (int(vp2[0]), int(vp2[1])), radius=1, color=(0, 0, 255))
            cv2.line(canvas, (int(vp1[0]), int(vp1[1])), (int(vp2[0]), int(vp2[1])), color=(0, 255, 255))
            cv2.imshow("Canvas", canvas)
        except Exception:
            ...
        # cv2.waitKey(1)
    cv2.waitKey(0)


def filter_vp(vp_data):
    filtered_vp_data = []

    current_frame = None
    cnt = 0
    for item in vp_data:
        if current_frame == item['frame_cnt']:
            cnt += 1
        else:
            cnt = 0
            current_frame = item['frame_cnt']

        if cnt < 10:
            filtered_vp_data.append(item)

    return filtered_vp_data


def get_best_vps(vp1, vp2, pp, args, session, bcp):
    if bcp:
        gt_data_path = os.path.join(args.bcp_path, 'ground_truth', session, 'gt_pairs.json')
        with open(gt_data_path, 'rb') as f:
            gt_data = json.load(f)
        distance_measurement = gt_data
    else:
        gt_data_path = os.path.join(args.bcs_path, 'dataset', session, 'gt_data.pkl')
        with open(gt_data_path, 'rb') as f:
            gt_data = pickle.load(f, encoding='latin-1', fix_imports=True)
        distance_measurement = gt_data["distanceMeasurement"]

    min_err = np.inf
    for vp1, vp2 in zip(vp1, vp2):
        projector = get_projector(vp1, vp2, pp)
        rel_errors, abs_errors = eval_pure_calibration(distance_measurement, projector)

        if np.mean(rel_errors) < min_err:
            min_err = np.mean(rel_errors)
            best_vp1 = vp1
            best_vp2 = vp2

    return best_vp1, best_vp2


def get_focal_kernel_voting(fs, scores, width=0.01, weight_exponent=1.5):
    min = np.ceil(np.min(fs))
    max = np.min([np.floor(np.max(fs)), 10000])
    n = max - min + 1
    n = n.astype(np.int)
    accum_space_x = np.linspace(start=min, stop=max, num=n)

    accum_space = np.sum(scores[:, np.newaxis]**weight_exponent * np.exp(-np.abs(fs[:, np.newaxis] - accum_space_x[np.newaxis, :]) / (width * n)), axis=0)

    # plt.plot(accum_space_x, accum_space)
    # plt.show()

    return accum_space_x[np.argmax(accum_space)]


def get_calib_vp(vp1, m, k, f, pp):
    # In this part we could choose arbitrary vp1 on the horizon, but we try to find point that actually is close to observed vp1s
    a, b, c = m, -1, k

    vp1_avg = np.median(vp1, axis=0)
    vp1_calib_x = (b * (b * vp1_avg[0] - a * vp1_avg[1]) - a * c)/(a**2 + b**2)
    vp1_calib_y = (a * (-b * vp1_avg[0] + a * vp1_avg[1]) - b * c)/(a**2 + b**2)

    aa = -(vp1_calib_x - pp[0])
    bb = -(vp1_calib_y - pp[1])
    cc = (vp1_calib_x - pp[0]) * pp[0] + (vp1_calib_y - pp[1]) * pp[1] - f ** 2
    vp2_calib = np.linalg.solve(np.array([[a, b], [aa, bb]]), np.array([-c, -cc]))

    return np.array([vp1_calib_x, vp1_calib_y]), vp2_calib


def project_to_horizon(vp1, vp2, m, k):
    a, b, c = m, -1, k

    vp1[:, 0] = (b * (b * vp1[:, 0] - a * vp1[:, 1]) - a * c)/(a**2 + b**2)
    vp1[:, 1] = (a * (-b * vp1[:, 0] + a * vp1[:, 1]) - b * c)/(a**2 + b**2)
    vp2[:, 0] = (b * (b * vp2[:, 0] - a * vp2[:, 1]) - a * c)/(a**2 + b**2)
    vp2[:, 1] = (a * (-b * vp2[:, 0] + a * vp2[:, 1]) - b * c)/(a**2 + b**2)

    return vp1, vp2


def export_calib_session(session, args, json_name, bcp=False):
    if bcp:
        vp_json_path = os.path.join(args.bcp_path, 'data', session, '{}.json'.format(json_name))
        calib_json_path = os.path.join(args.bcp_path, 'results', session, 'system_{}_{}c.json'.format(json_name, args.conf))
    else:
        vp_json_path = os.path.join(args.bcs_path, 'dataset', session, '{}.json'.format(json_name))
        calib_json_path = os.path.join(args.bcs_path, 'results', session, 'system_{}_{}c.json'.format(json_name, args.conf))

    if session == 'S09' or session == 'S10' or session == 'S11':
        pp = np.array([640.5, 400.5])
    else:
        pp = np.array([960.5, 540.5])

    print("Starting for session {}".format(session))

    with open(vp_json_path, 'r') as f:
        vp_data = json.load(f)

    if bcp:
        vp_data = filter_vp(vp_data)

    if 'pred_vars' in vp_data[0].keys():
        pred_vars = np.array([item['pred_vars'] for item in vp_data if item['score'] > args.conf])
        pred_vars_vp1 = pred_vars[:, :, 0]
        pred_vars_vp2 = pred_vars[:, :, 1]
        best_scales_vp1 = np.argmin(pred_vars_vp1, axis=-1)
        best_scales_vp2 = np.argmin(pred_vars_vp2, axis=-1)
        _, counts_vp1 = np.unique(best_scales_vp1, return_counts=True)
        _, counts_vp2 = np.unique(best_scales_vp2, return_counts=True)

        # print("VP1 best scale counts: ", counts_vp1)
        # print("VP2 best scale counts: ", counts_vp2)
    else:
        counts_vp1 = np.zeros(4)
        counts_vp2 = np.zeros(4)




    # if 'vp1_var' in vp_data[0].keys():
    #     scores = np.array([item['score'] / (item['vp1_var'] * item['vp2_var']) for item in vp_data])
    #     vp1 = np.array([item['vp1'] for item in vp_data])
    #     vp2 = np.array([item['vp2'] for item in vp_data])
    #     f = np.sqrt(-np.sum((vp1 - pp[np.newaxis, :]) * (vp2 - pp[np.newaxis, :]), axis=1))

    scores = np.array([item['score'] for item in vp_data if item['score'] > args.conf])
    vp1 = np.array([item['vp1'] for item in vp_data if item['score'] > args.conf])
    vp2 = np.array([item['vp2'] for item in vp_data if item['score'] > args.conf])

    f = np.sqrt(-np.sum((vp1 - pp[np.newaxis, :]) * (vp2 - pp[np.newaxis, :]), axis=1))

    scores = scores[~np.isnan(f)]
    vp1 = vp1[~np.isnan(f)]
    vp2 = vp2[~np.isnan(f)]
    f = f[~np.isnan(f)]

    if args.debug:
        show_vps(np.array(vp1), np.array(vp2), session)

    med_f = np.nanmedian(f)

    m = (vp1[:, 1] - vp2[:, 1]) / (vp1[:, 0] - vp2[:, 0])
    b1 = vp1[:, 1] - m * vp1[:, 0]
    b2 = vp2[:, 1] - m * vp2[:, 0]

    med_m = np.nanmedian(m)
    med_k = np.nanmedian(np.concatenate([b1, b2]))

    vp1_calib, vp2_calib = get_calib_vp(vp1, med_m, med_k, med_f, pp)

    if args.debug:
        show_vps(np.array([vp1_calib]), np.array([vp2_calib]), session)

    lp1 = np.array([item['lp1'] for item in vp_data if "lp1" in item and item['lp1'] is not None and item['score'] > args.conf])
    lp2 = np.array([item['lp2'] for item in vp_data if "lp2" in item and item['lp2'] is not None and item['score'] > args.conf])

    if len(lp1) == 0:
        calib_dict = get_calib_dict(vp1_calib, vp2_calib, pp)
        save(calib_json_path, calib_dict)

    else:
        dists = []
        projector = get_projector(vp1_calib, vp2_calib, pp)
        for p1, p2 in zip(lp1, lp2):
            ip1 = projector(p1)
            ip2 = projector(p2)

            dists.append(np.linalg.norm(ip1 - ip2))

        med_dist = np.nanmedian(dists)
        scale = 0.52 / med_dist
        calib_dict = get_calib_dict(vp1_calib, vp2_calib, pp, scale=scale)
        save(calib_json_path, calib_dict)

    return counts_vp1, counts_vp2


def export_calib():
    args = parse_command_line()

    for json_name in args.json_names:
        print("Running for: ", json_name)

        # total_counts_vp1 = np.zeros(4)
        # total_counts_vp2 = np.zeros(4)

        sessions = sorted(os.listdir(os.path.join(args.bcs_path, 'dataset')))[12:]
        for session in sessions:
            cnt_vp1, cnt_vp2 = export_calib_session(session, args, json_name)
            # total_counts_vp1 += cnt_vp1
            # total_counts_vp2 += cnt_vp2

        sessions = sorted(os.listdir(os.path.join(args.bcp_path, 'data')))
        for session in sessions:
            cnt_vp1, cnt_vp2 = export_calib_session(session, args, json_name, bcp=True)
            # total_counts_vp1 += cnt_vp1
            # total_counts_vp2 += cnt_vp2

        # print("Finished: ", json_name)
        # print("Total vp1 cnt: ", total_counts_vp1)
        # print("Total vp2 cnt: ", total_counts_vp2)
        # print('*' * 40)




if __name__ == '__main__':
    export_calib()