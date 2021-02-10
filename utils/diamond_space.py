import numpy as np

def diamond_coords_from_original(p, d):
    if len(p) == 2:
        p = np.array([p[0], p[1], 1])
    else:
        p = np.array(p)
    p_diamond = np.array([- d**2 * p[2], -d * p[0], np.sign(p[0]*p[1]) * p[0] + p[1] + np.sign(p[1]) * d * p[2]])
    return p_diamond[:2] / p_diamond[2]


def original_coords_from_diamond(p, d):
    if len(p) == 2:
        p = np.array([p[0], p[1], 1])
    else:
        p = np.array(p)

    p_original = np.array([d * p[1], np.sign(p[0]) * d * p[0] + np.sign(p[1]) * d * p[1] - d**2 * p[2], p[0]])
    return p_original[:2] / p_original[2]