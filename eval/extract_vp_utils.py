import json

import cv2
import numpy as np
from matplotlib import pyplot as plt


from object_detection.detect_utils import get_mask_frame
from utils.diamond_space import process_heatmaps


def is_left(a, b, p):
    ret = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
    return ret < 0


def is_right(a, b, p):
    return not is_left(a, b, p)


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(l1, l2):
    d = l1[0] * l2[1] - l1[1] * l2[0]
    d_x = l1[2] * l2[1] - l1[1] * l2[2]
    d_y = l1[0] * l2[2] - l1[2] * l2[0]
    if d != 0:
        x = d_x / d
        y = d_y / d
        return np.float32(x), np.float32(y)
    else:
        return False


def get_vp3(vp1, vp2, pp):
    focal = np.sqrt(- np.dot(vp1 - pp, vp2 - pp))

    vp1_w = np.concatenate((vp1, [focal]))
    vp2_w = np.concatenate((vp2, [focal]))
    pp_w = np.concatenate((pp, [0]))

    vp3_w = np.cross(vp1_w-pp_w, vp2_w-pp_w)
    vp3 = vp3_w[0:2]/vp3_w[2] * focal + pp_w[0:2]
    return vp3

def tangent_point_poly(p, V):
    left_idx = 0
    right_idx = 0
    p = [np.float64(x) for x in p]
    n = len(V)
    for i in range(1, n):
        if is_left(p, V[left_idx], V[i]):
            left_idx = i
        if not is_left(p, V[right_idx], V[i]):
            right_idx = i
    # if p[1] > self.im_h:
    #     return V[left_idx], V[right_idx]
    return V[right_idx], V[left_idx]


def get_pts_from_mask(mask, vp1, vp2):
    countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    hull = cv2.convexHull(countours[0])
    pts = hull[:, 0, :]
    idx1, idx2 = find_cornerpts(vp1, pts)
    idx3, idx4 = find_cornerpts(vp2, pts)
    pts = pts[[idx1, idx2, idx3, idx4]]

    return [pts[0], pts[3], pts[2], pts[1]]


def find_cornerpts(VP, pts):
    pts = np.array(pts)
    for p1 in range(len(pts)):
        bad = False
        for idx in range(len(pts)):
            if (pts[idx] != pts[p1]).any() and is_right(VP, pts[p1], pts[idx]):
                bad = True
                break
        if not bad:
            break

    for p2 in range(len(pts)):
        bad = False
        for idx in range(len(pts)):
            if (pts[idx] != pts[p2]).any() and not is_right(VP, pts[p2], pts[idx]):
                bad = True
                break
        if not bad:
            break

    return p1, p2


def get_transform_matrix(vp1, vp2, image, im_w, im_h, pts=None, enforce_vp1=True, vp_top=None):
    if pts is None:
        pts = [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]]

    vp1p1, vp1p2 = find_cornerpts(vp1, pts)
    vp2p1, vp2p2 = find_cornerpts(vp2, pts)

    # right side
    vp1l1 = line(vp1, pts[vp1p1])
    # left side
    vp1l2 = line(vp1, pts[vp1p2])
    # right side
    vp2l1 = line(vp2, pts[vp2p1])
    # left side
    vp2l2 = line(vp2, pts[vp2p2])

    t_dpts = [[0, 0], [0, im_h], [im_w, im_h], [im_w, 0]]

    ipts = []
    ipts.append(intersection(vp1l1, vp2l1))
    ipts.append(intersection(vp1l2, vp2l1))
    ipts.append(intersection(vp1l1, vp2l2))
    ipts.append(intersection(vp1l2, vp2l2))

    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for p in ipts:
    #     image = cv2.circle(image, p, 40, (0, 0, 255), thickness=3)
    # cv2.imshow("Mask with pts", image)
    # cv2.waitKey(0)

    if enforce_vp1:
        if vp1[1] > im_h:
            t_dpts = [[im_w, 0], [im_w, im_h], [0, im_h], [0, 0]]

        t_ipts = np.zeros((4,2), dtype=np.float32)
        t_pts = np.array(t_dpts, np.float32)

        if ipts[0][1] < ipts[2][1]:
            t_ipts[0, :] = ipts[0]
            t_ipts[1, :] = ipts[2]
        else:
            t_ipts[0, :] = ipts[2]
            t_ipts[1, :] = ipts[0]
        if ipts[1][1] < ipts[3][1]:
            t_ipts[3, :] = ipts[1]
            t_ipts[2, :] = ipts[3]
        else:
            t_ipts[3, :] = ipts[3]
            t_ipts[2, :] = ipts[1]

    if vp_top is not None:
        t_ipts = np.roll(t_ipts, -1, axis=0)
        for roll in range(4):
            t_ipts = np.roll(t_ipts, 1, axis=0)
            M = cv2.getPerspectiveTransform(t_ipts, t_pts)
            vp_top_t = cv2.perspectiveTransform(np.array([[vp_top]]), M)
            if vp_top_t[0, 0, 1] < 0:
                return cv2.getPerspectiveTransform(t_ipts, t_pts), cv2.getPerspectiveTransform(t_pts, t_ipts)
    return cv2.getPerspectiveTransform(t_ipts, t_pts), cv2.getPerspectiveTransform(t_pts, t_ipts)


def get_lp(img, mask):
    filtered_img = cv2.bilateralFilter(img, 9, 200, 50)
    # filtered_img = cv2.bilateralFilter(filtered_img, 9, 150, 50)
    gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

    gray_mean = np.mean(gray_img[mask > 127])
    cv2.waitKey(0)
    mser = cv2.MSER_create(_delta=3, _max_variation=0.1)

    # regions, bboxes = mser.detectRegions(cv2.edgePreservingFilter(img))
    regions, bboxes = mser.detectRegions(filtered_img)
    # regions, bboxes = mser.detectRegions(cv2.blur(img, (11, 11)))
    # regions, bboxes = mser.detectRegions(gray_img)
    # print(len(regions))

    # vis_img = np.copy(img)

    best_bbox_area = 0
    best_region = None

    for region in regions:
        if np.mean(gray_img[region[:, 1], region[:, 0]]) < gray_mean:
            continue

        x_min = np.min(region[:, 0])
        x_max = np.max(region[:, 0])
        y_min = np.min(region[:, 1])
        y_max = np.max(region[:, 1])

        if (x_max - x_min)/(y_max - y_min) < 1.5:
            continue

        bbox_area = (x_max - x_min) * (y_max - y_min)

        if np.abs(bbox_area - len(region)) > 0.1 * len(region):
            continue

        if bbox_area > best_bbox_area:
            best_bbox_area = bbox_area
            best_region = region

        # vis_img[region[:, 1], region[:, 0]] = np.array([0, 255, 0], dtype=np.uint8)
        # vis_img = cv2.rectangle(vis_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255))
        # cv2.imshow("Reg", vis_img)
        # cv2.waitKey(0)

    if best_region is None:
        return None, None

    x_min = np.min(best_region[:, 0])
    x_max = np.max(best_region[:, 0])
    # y_min = np.min(best_region[:, 1])
    # y_max = np.max(best_region[:, 1])

    # vis_img[best_region[:, 1], best_region[:, 0]] = np.array([0, 255, 0], dtype=np.uint8)
    # vis_img = cv2.rectangle(vis_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 255))
    # cv2.imshow("Reg", vis_img)
    # cv2.waitKey(0)

    return x_min, x_max


def save(json_path, detection_list):
    with open(json_path, 'w') as f:
        json.dump(detection_list, f)


class BatchVPDetectorBase():
    def __init__(self, model, args):
        self.model = model
        self.input_size = args.input_size
        self.batch_size = args.batch_size_eval
        self.debug = args.debug

        self.q = []
        self.output_list = []

    def process(self, frame, box, score, **kwargs):
        self.last_frame = frame

        x_min = int(frame.shape[1] * box[1])
        y_min = int(frame.shape[0] * box[0])
        x_max = int(frame.shape[1] * box[3] + 1)
        y_max = int(frame.shape[0] * box[2] + 1)

        box_center = np.array([x_min + x_max, y_min + y_max]) / 2
        box_scale = np.array([x_max - x_min, y_max - y_min]) / 2
        try:
            orig_car_img = frame[y_min:y_max, x_min:x_max, :]
            car_img = cv2.resize(orig_car_img, (self.input_size, self.input_size), cv2.INTER_CUBIC)
        except Exception as e:
            print("Caught exception:")
            print(str(e))
            print("Ignoring box")
            return

        vp_box = [y_min, x_min, y_max, x_max]

        item = {'orig_car_img': orig_car_img, 'car_img': car_img, 'box': box, 'box_center': box_center, 'box_scale': box_scale, 'vp_box': vp_box, 'score': score, **kwargs}
        self.q.append(item)
        if len(self.q) == self.batch_size:
            self.predict()

    def finalize(self):
        if len(self.q) > 0:
            self.predict()

    def get_lp_from_mask(self, item):
        item['lp1'] = None
        item['lp2'] = None

        mask_frame = get_mask_frame(item['box'], self.last_frame, np.array(item['mask']))
        _, mask_frame = cv2.threshold(mask_frame, 0.5, 255, cv2.THRESH_BINARY)

        del item['mask']

        image = np.zeros_like(self.last_frame)
        x_min = item['vp_box'][1]
        y_min = item['vp_box'][0]
        x_max = item['vp_box'][3]
        y_max = item['vp_box'][2]
        image[y_min:y_max, x_min:x_max, :] = item['orig_car_img']
        image = cv2.bitwise_and(image, image, mask=mask_frame.astype(np.uint8))

        vp1 = np.array(item['vp1'])
        vp2 = np.array(item['vp2'])
        pp = np.array([self.last_frame.shape[1] / 2 + 0.5, self.last_frame.shape[0] / 2 + 0.5])
        vp3 = get_vp3(vp1, vp2, pp)

        if np.isnan(vp3).any():
            return item

        mask_pts = get_pts_from_mask(mask_frame.astype(np.uint8), vp3, vp2)

        # for p in mask_pts:
        #     image = cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), thickness=-1)
        # cv2.imshow("Img with mask pts", image)

        M, IM = get_transform_matrix(vp3, vp2, image, 300, 300, pts=mask_pts, vp_top=vp1)

        warped_image = cv2.warpPerspective(image, M, (300, 300))
        warped_mask = cv2.warpPerspective(mask_frame, M, (300, 300))

        # cv2.imshow("warped image", warped_image)
        # cv2.waitKey(0)

        lp_warped_x1, lp_warped_x2 = get_lp(warped_image[150:], warped_mask[150:])

        if lp_warped_x1 is None or lp_warped_x2 is None:
            return item

        lps_warped = np.array([[[lp_warped_x1, 300]], [[lp_warped_x2, 300]]], dtype=np.float32)
        lps = cv2.perspectiveTransform(lps_warped, IM)

        item['lp1'] = lps[0, 0, :].tolist()
        item['lp2'] = lps[1, 0, :].tolist()

        # image = cv2.circle(image, (int(item['lp1'][0]), int(item['lp1'][1])), 3, (0, 0, 255), thickness=-1)
        # image = cv2.circle(image, (int(item['lp2'][0]), int(item['lp2'][1])), 3, (0, 0, 255), thickness=-1)
        # cv2.imshow("Img with mask pts", image)
        # cv2.waitKey(0)

        return item


    def draw_debug(self, item):
        x_min = item['vp_box'][1]
        y_min = item['vp_box'][0]
        x_max = item['vp_box'][3]
        y_max = item['vp_box'][2]

        frame = cv2.rectangle(np.copy(self.last_frame), (x_min, y_min), (x_max, y_max), (0, 0, 255))
        try:
            box_center = (int(item['box_center'][0]), int(item['box_center'][1]))
            vp1 = (int(item['vp1'][0]), int(item['vp1'][1]))
            vp2 = (int(item['vp2'][0]), int(item['vp2'][1]))
            frame = cv2.line(frame, vp1, box_center, (0, 255, 0))
            frame = cv2.line(frame, vp2, box_center, (0, 255, 0))
        except Exception:
            ...

        cv2.imshow("VPs debug", frame)
        cv2.waitKey(1)


class BatchVPDetectorReg(BatchVPDetectorBase):
    def predict(self):
        cars = [item['car_img'] for item in self.q]
        preds = self.model.predict(np.array(cars) / 255)
        if isinstance(preds, list):
            preds = preds[-1]
        for pred, item in zip(preds, self.q):
            vp1_box = pred[:2]
            vp2_box = pred[2:]

            vp1 = item['box_scale'] * vp1_box + item['box_center']
            vp2 = item['box_scale'] * vp2_box + item['box_center']

            item['box_scale'] = item['box_scale'].tolist()
            item['box_center'] = item['box_center'].tolist()
            item['vp1_box'] = vp1_box.tolist()
            item['vp2_box'] = vp2_box.tolist()
            item['vp1'] = vp1.tolist()
            item['vp2'] = vp2.tolist()

            if 'mask' in item.keys():
                item = self.get_lp_from_mask(item)

            del item['car_img']
            del item['orig_car_img']

            self.output_list.append(item)

        if self.debug:
            self.draw_debug(item)

        self.q = []


class BatchVPDetectorHeatmap(BatchVPDetectorBase):
    def __init__(self, model, args, scales):
        self.scales = scales
        super(BatchVPDetectorHeatmap, self).__init__(model, args)

    def predict(self):
        cars = [item['car_img'] for item in self.q]
        preds = self.model.predict(np.array(cars) / 255)
        pred_vps, pred_vars = process_heatmaps(preds[-1], self.scales)
        for pred_vp, pred_var, item in zip(pred_vps, pred_vars, self.q):
            vp1_var = pred_var[:, 0]
            vp2_var = pred_var[:, 1]
            vp1_var_idx = np.argmin(vp1_var, axis=-1)
            vp2_var_idx = np.argmin(vp2_var, axis=-1)

            vp1_box = pred_vp[vp1_var_idx, :2]
            vp2_box = pred_vp[vp2_var_idx, 2:]

            vp1 = item['box_scale'] * vp1_box + item['box_center']
            vp2 = item['box_scale'] * vp2_box + item['box_center']

            item['vp1_var'] = vp1_var[vp1_var_idx]
            item['vp2_var'] = vp1_var[vp2_var_idx]

            item['pred_vps'] = pred_vp.tolist()
            item['pred_vars'] = pred_var.tolist()

            item['box_scale'] = item['box_scale'].tolist()
            item['box_center'] = item['box_center'].tolist()
            item['vp1_box'] = vp1_box.tolist()
            item['vp2_box'] = vp2_box.tolist()
            item['vp1'] = vp1.tolist()
            item['vp2'] = vp2.tolist()

            if 'mask' in item.keys():
                item = self.get_lp_from_mask(item)

            del item['car_img']
            del item['orig_car_img']

            self.output_list.append(item)

        if self.debug:
            self.draw_debug(item)

        self.q = []


def filter_boxes_bcp(boxes, scores, frame, prev_edge, masks=None):
    filtered_boxes = []
    filtered_scores = []
    filtered_masks = []

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(frame, 100, 200)
    if prev_edge is None:
        edge_diff = edge
    else:
        edge_diff = np.abs(edge - prev_edge)

    for i, box in enumerate(boxes):
        x_min = int(frame.shape[1] * box[1])
        y_min = int(frame.shape[0] * box[0])
        x_max = int(frame.shape[1] * box[3] + 1)
        y_max = int(frame.shape[0] * box[2] + 1)

        edge_box = edge_diff[y_min: y_max, x_min: x_max]
        if np.mean(edge_box) > 5.0:
            filtered_boxes.append(box)
            filtered_scores.append(scores[i])
            if masks is not None:
                filtered_masks.append(masks[i])
            if len(filtered_scores) >= 10:
                break

    prev_edge = edge

    if masks is None:
        filtered_masks = None

    return filtered_boxes, filtered_scores, filtered_masks, prev_edge