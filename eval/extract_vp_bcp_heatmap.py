import os

from eval.extract_vp_utils import save, BatchVPDetectorHeatmap, filter_boxes_bcp
from models.hourglass import parse_command_line, load_model

import datetime
import json
import time

import cv2
import numpy as np
from object_detection.detect_utils import show_debug

from utils.gpu import set_gpus


def detect_session(detector, model_dir_name, data_path, session, args, scales):
    batch_vp_detector = BatchVPDetectorHeatmap(detector, args, scales)

    print("Starting vp detection for ", session)

    if args.mask:
        output_json_name = 'VPout_{}_r{}_mask.json'.format(model_dir_name, args.resume)
        json_path = os.path.join(data_path, 'data', session, 'detections_mask.json')
    else:
        output_json_name = 'VPout_{}_r{}.json'.format(model_dir_name, args.resume)
        json_path = os.path.join(data_path, 'data', session, 'detections.json')

    with open(json_path, 'r') as f:
        detection_data = json.load(f)

    detection_data = detection_data[:args.max_frames]

    output_json_path = os.path.join(data_path, 'data', session, output_json_name)

    total_box_count = sum([len(item['boxes']) for item in detection_data])
    print("Loaded {} bounding boxes for {} frames".format(total_box_count, len(detection_data)))
    box_cnt = 0

    start_time = time.time()

    # running_average_frame = np.zeros([1080, 1920, 2], dtype=np.float32)
    prev_edge = None
    masks = None

    for detection_cnt, detection in enumerate(detection_data):
        frame_filename = detection['filename']
        frame = cv2.imread(os.path.join(data_path, 'frames', session, frame_filename))
        frame_cnt = detection['frame_cnt']

        boxes = detection['boxes']
        scores = detection['scores']
        del detection['boxes']
        del detection['scores']

        box_cnt += len(boxes)

        if args.mask:
            masks = detection['masks']
            del detection['masks']

        boxes, scores, masks, prev_edge = filter_boxes_bcp(boxes, scores, frame, prev_edge, masks=masks)

        if args.debug:
            show_debug(np.copy(frame), boxes)

        for i in range(len(boxes)):
            if args.mask:
                batch_vp_detector.process(frame, boxes[i], scores[i], frame_cnt=frame_cnt, frame_filename=frame_filename, mask=masks[i])
            else:
                batch_vp_detector.process(frame, boxes[i], scores[i], frame_cnt=frame_cnt, frame_filename=frame_filename)

        if args.dump_every != 0 and detection_cnt % args.dump_every == 0:
            print("Saving at detection ", detection_cnt)
            save(output_json_path, batch_vp_detector.output_list)

        remaining_seconds = (time.time() - start_time) / (box_cnt + 1) * (total_box_count - box_cnt)
        print('{} : {}, Box: {} / {}, ETA: {}'.format(frame_cnt, frame_filename, box_cnt, total_box_count, datetime.timedelta(seconds=remaining_seconds)))

    batch_vp_detector.finalize()
    print("Saving at box ", box_cnt)
    save(output_json_path, batch_vp_detector.output_list)
    print("Finished session: {} with {} boxes".format(session, total_box_count))


def detect():
    args = parse_command_line()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_gpus()

    model, scales, model_dir_name, _ = load_model(args)

    data_path = args.path
    sessions = sorted(os.listdir(os.path.join(data_path, 'data')))
    for session in sessions:
        detect_session(model, model_dir_name, data_path, session, args, scales)


if __name__ == '__main__':
    detect()