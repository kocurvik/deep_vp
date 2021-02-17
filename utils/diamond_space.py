import numpy as np

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


def heatmap_to_vp(heatmap, scale):
    res = heatmap.shape[0]
    vp_heatmap = np.array(np.unravel_index(heatmap.argmax(), heatmap.shape))
    Rinv = np.linalg.inv(np.array([[1, -1], [1, 1]]))
    vp_diamond = Rinv @ (2/res * vp_heatmap - 1.0)
    vp_scaled = original_coords_from_diamond(vp_diamond, 1.0)
    vp = vp_scaled / scale

    return vp