import numpy as np


def get_focal(vp1, vp2, pp):
    return np.sqrt(-np.dot(vp1[0:2]-pp[0:2], vp2[0:2]-pp[0:2]))


def diamond_coords_from_original(p, d):
    if len(p) == 2:
        p = np.array([p[0], p[1], 1])
    else:
        p = np.array(p)
    p_diamond = np.array([- d**2 * p[2], -d * p[0], np.sign(p[0]*p[1]) * p[0] + p[1] + np.sign(p[1]) * d * p[2]])
    return p_diamond[:2] / p_diamond[2]


def original_coords_from_diamond(p, d, return_homogenous=False):
    if len(p) == 2:
        p = np.array([p[0], p[1], 1])
    else:
        p = np.array(p)

    p_original = np.array([d * p[1], np.sign(p[0]) * d * p[0] + np.sign(p[1]) * d * p[1] - d**2 * p[2], p[0]])
    if return_homogenous:
        return p_original

    return p_original[:2] / p_original[2]


def heatmap_to_vp(vp_heatmap, res, scale=1.0):
    Rinv = np.linalg.inv(np.array([[1, -1], [1, 1]]))
    vp_diamond = Rinv @ (2 / (res - 1) * vp_heatmap - 1.0)
    vp_scaled = original_coords_from_diamond(vp_diamond, 1.0)
    vp = vp_scaled / scale
    return vp


def vp_to_heatmap(vp, res, scale=1.0):
    vp_scaled = vp * scale
    vp_diamond = diamond_coords_from_original(vp_scaled, 1.0)
    R = np.array([[1, -1], [1, 1]])
    vp_heatmap = ((R @ vp_diamond.T) + 1.0) * (res - 1) / 2
    return vp_heatmap


def heatmap_to_orig(res, scale=1.0):
    heatmap_orig = np.empty([res, res, 2], dtype=np.float)
    for i in range(res):
        for j in range(res):
            heatmap_orig[i, j] = heatmap_to_vp(np.array([i, j], dtype=np.float), res, scale)

    return heatmap_orig

def process_heatmap(heatmap, scale):
    heatmap_orig = heatmap_to_orig(heatmap.shape[0], scale=scale)

    heatmap_orig_x = heatmap_orig[:, :, 0][~np.logical_or(np.isinf(heatmap_orig[:, :, 0]), np.isnan(heatmap_orig[:, :, 0]))]
    heatmap_orig_y = heatmap_orig[:, :, 1][~np.logical_or(np.isinf(heatmap_orig[:, :, 1]), np.isnan(heatmap_orig[:, :, 1]))]

    weights_x = heatmap[~np.logical_or(np.isinf(heatmap_orig[:, :, 0]), np.isnan(heatmap_orig[:, :, 0]))]
    weights_y = heatmap[~np.logical_or(np.isinf(heatmap_orig[:, :, 1]), np.isnan(heatmap_orig[:, :, 1]))]

    vp_x_avg = np.average(heatmap_orig_x, weights=weights_x)
    vp_y_avg = np.average(heatmap_orig_y, weights=weights_y)
    vp_x_std = np.sqrt(np.average((heatmap_orig_x - vp_x_avg) ** 2, weights=weights_x))
    vp_y_std = np.sqrt(np.average((heatmap_orig_y - vp_y_avg) ** 2, weights=weights_y))

    return np.array([vp_x_avg, vp_y_avg]), np.array([vp_x_std, vp_y_std])


def process_heatmap_old(heatmap, scale):
    max_heatmap = heatmap.max()
    vp_heatmap_max = np.array(np.unravel_index(heatmap.argmax(), heatmap.shape)) + 0.5
    vp_max = heatmap_to_vp(vp_heatmap_max, heatmap.shape[0], scale=scale)

    vps_heatmap_plausible = np.vstack(np.where(heatmap > 0.8 * max_heatmap)).T
    dists = [np.linalg.norm(vp_max - heatmap_to_vp(vp, heatmap.shape[0], scale=scale)) for vp in vps_heatmap_plausible]
    mean_dist = np.mean(dists) / np.linalg.norm(vp_max)

    return vp_max, mean_dist

def process_heatmaps(heatmaps, scales):

    vps = np.empty([heatmaps.shape[0], len(scales), 4])
    dists = np.empty([heatmaps.shape[0], len(scales), 2])


    for i in range(heatmaps.shape[0]):
        for j, scale in enumerate(scales):
            heatmap_vp1 = heatmaps[i, :, :, j]
            heatmap_vp2 = heatmaps[i, :, :, j + len(scales)]

            vp1, vp1_dist = process_heatmap_old(heatmap_vp1, scale)
            vp2, vp2_dist = process_heatmap_old(heatmap_vp2, scale)

            vps[i, j, :2] = vp1
            vps[i, j, 2:] = vp2

            if np.isnan(vp1).any() or np.isinf(vp1).any():
                dists[i, j, 0] = np.inf
            else:
                dists[i, j, 0] = vp1_dist
            if np.isnan(vp2).any() or np.isinf(vp2).any():
                dists[i, j, 1] = np.inf
            else:
                dists[i, j, 1] = vp2_dist

    return vps, dists
