import argparse
import json
import os

import numpy as np
import cv2
from utils.diamond_space import get_focal


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-c', '--conf', type=float, default=0.5)
    parser.add_argument('path')
    parser.add_argument('json_names', metavar='N', nargs='+')

    args = parser.parse_args()
    return args


def get_calib_dict(vp1, vp2):
    return {'cars': [], 'camera_calibration': {"pp": [960.5, 540.5], "scale": 1.0, "vp1": [vp1[0], vp1[1]], "vp2": [vp2[0], vp2[1]]}}


def show_vps(vp1s, vp2s, data_path, session):
    canvas = np.zeros([1080, 1920, 3])
    zero = np.array([1920 // 2 - 1920 // 20, 1080 // 2 - 1080 // 20])
    canvas = cv2.rectangle(canvas, (1920 // 2 - 1920 //20, 1080 // 2 - 1080 // 20), (1920 // 2 + 1920 // 20, 1080 // 2 + 1080 // 20), (255, 255, 255))


    compare_calib_json_path = os.path.join(data_path, 'results', session, 'system_SochorCVIU_ManualCalib_ManualScale.json')
    with open(compare_calib_json_path, 'r') as f:
        compare_calib = json.load(f)
    compare_vp1 = compare_calib['camera_calibration']['vp1']
    compare_vp2 = compare_calib['camera_calibration']['vp2']
    compare_vp1 = np.array(compare_vp1) / 10 + zero
    compare_vp2 = np.array(compare_vp2) / 10 + zero

    cv2.circle(canvas, (int(compare_vp1[0]), int(compare_vp1[1])), radius=3, color=(0, 255, 0))
    cv2.circle(canvas, (int(compare_vp2[0]), int(compare_vp2[1])), radius=3, color=(0, 0, 255))
    cv2.line(canvas, (int(compare_vp1[0]), int(compare_vp1[1])), (int(compare_vp2[0]), int(compare_vp2[1])), color=(255, 0, 0), thickness=3)

    vp1s = vp1s / 10 + zero
    vp2s = vp2s / 10 + zero

    for vp1, vp2 in zip(vp1s, vp2s):
        cv2.circle(canvas, (int(vp1[0]), int(vp1[1])), radius=1, color=(0, 255, 0))
        cv2.circle(canvas, (int(vp2[0]), int(vp2[1])), radius=1, color=(0, 0, 255))
        cv2.line(canvas, (int(vp1[0]), int(vp1[1])), (int(vp2[0]), int(vp2[1])), color=(0, 255, 255))
        cv2.imshow("Canvas", canvas)
        cv2.waitKey(0)



def export_calib_session(session, args, json_name):
    vp_json_path = os.path.join(args.path, 'dataset', session, '{}.json'.format(json_name))
    calib_json_path = os.path.join(args.path, 'results', session, 'system_{}_{}c.json'.format(json_name, args.conf))

    pp = np.array([960.5, 540.5])

    print("Starting for session {}".format(session))

    with open(vp_json_path, 'r') as f:
        vp_data = json.load(f)

    vp1 = np.array([item['vp1'] for item in vp_data if item['score'] > args.conf])
    vp2 = np.array([item['vp2'] for item in vp_data if item['score'] > args.conf])
    f = np.sqrt(-np.sum((vp1 - pp[np.newaxis, :]) * (vp2 - pp[np.newaxis, :]), axis=1))

    if args.debug:
        show_vps(vp1, vp2, args.path, session)

    vp1 = vp1[~np.isnan(f)]
    vp2 = vp2[~np.isnan(f)]
    f = f[~np.isnan(f)]

    m = (vp1[:, 1] - vp2[:, 1]) / (vp1[:, 0] - vp2[:, 0])
    b1 = vp1[:, 1] - m * vp1[:, 0]
    b2 = vp2[:, 1] - m * vp2[:, 0]

    med_m = np.nanmedian(m)
    med_b = np.nanmedian(np.concatenate([b1, b2]))
    med_f = np.nanmedian(f)

    a, b, c = med_m, -1, med_b

    print("Median horizon line: y = {} * x + {}".format(med_m, med_b))
    print("Median horizon line: {} * x + {} * y + {} = 0".format(a, b, c))

    # In this part we could choose arbitrary vp1 on the horizon, but we try to find point that actually is close to observed vp1s
    vp1_avg = np.median(vp1, axis=0)
    vp1_calib_x = (b * (b * vp1_avg[0] - a * vp1_avg[1]) - a * c)/(a**2 + b**2)
    vp1_calib_y = (a * (-b * vp1_avg[0] + a * vp1_avg[1]) - b * c)/(a**2 + b**2)

    aa = -(vp1_calib_x - pp[0])
    bb = -(vp1_calib_y - pp[1])
    cc = (vp1_calib_x - pp[0]) * pp[0] + (vp1_calib_y - pp[1]) * pp[1] - med_f ** 2

    vp2_calib = np.linalg.solve(np.array([[a, b], [aa, bb]]), np.array([-c, -cc]))
    calib_dict = get_calib_dict([vp1_calib_x, vp1_calib_y], vp2_calib)

    # vp2_avg = np.median(vp2, axis=1)
    # vp2_calib_x = (b * (b * vp2_avg[0] - a * vp2_avg[1]) - a * c)/(a**2 + b**2)
    # vp2_calib_y = (a * (-b * vp2_avg[0] + a * vp2_avg[1]) - b * c)/(a**2 + b**2)
    # calib_dict = get_calib_dict([vp1_calib_x, vp1_calib_y], [vp2_calib_x, vp2_calib_y])

    with open(calib_json_path, 'w') as f:
        json.dump(calib_dict, f)

def export_calib():
    args = parse_command_line()

    for json_name in args.json_names:

        print("Running for: ", json_name)
        data_path = args.path
        sessions = os.listdir(os.path.join(data_path, 'dataset'))
        for session in sessions:
            export_calib_session(session, args, json_name)


if __name__ == '__main__':
    export_calib()