import argparse
import json
import os

import numpy as np
from utils.diamond_space import get_focal


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', type=float, default=0.5)
    parser.add_argument('json_name')
    parser.add_argument('path')

    args = parser.parse_args()
    return args


def get_calib_dict(vp1, vp2):
    return {'cars': [], 'camera_calibration': {"pp": [960.5, 540.5], "scale": 1.0, "vp1": [vp1[0], vp1[1]], "vp2": [vp2[0], vp2[1]]}}


def export_calib_session(session, args):
    vp_json_path = os.path.join(args.path, 'dataset', session, '{}.json'.format(args.json_name))
    calib_json_path = os.path.join(args.path, 'results', session, 'system_{}_{}c.json'.format(args.json_name, args.conf))

    pp = np.array([960.5, 540.5])

    print("Starting for session {}".format(session))

    with open(vp_json_path, 'r') as f:
        vp_data = json.load(f)

    vp1 = np.array([item['vp1'] for item in vp_data if item['score'] > args.conf])
    vp2 = np.array([item['vp2'] for item in vp_data if item['score'] > args.conf])
    f = np.sqrt(-np.sum((vp1 - pp[np.newaxis, :]) * (vp2 - pp[np.newaxis, :]), axis=1))

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

    with open(calib_json_path, 'w') as f:
        json.dump(calib_dict, f)

def export_calib():
    args = parse_command_line()

    data_path = args.path
    sessions = os.listdir(os.path.join(data_path, 'dataset'))
    for session in sessions:
        export_calib_session(session, args)


if __name__ == '__main__':
    export_calib()