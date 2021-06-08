import os
import datetime
import json
import time

import cv2
import numpy as np

from object_detection.detect_utils import show_debug
from eval.extract_vp_utils import save, BatchVPDetectorHeatmap
from models.hourglass import parse_command_line, load_model
from utils.gpu import set_gpus


def detect_session(detector, model_dir_name, data_path, session, args, scales):
    batch_vp_detector = BatchVPDetectorHeatmap(detector, args, scales)

    print("Starting object detection for ", session)
    cap = cv2.VideoCapture(os.path.join(data_path, 'dataset', session, 'video.avi'))
    # DO NOT REMOVE OTHERWISE FRAMES WILL NOT SYNC
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print("Video loaded!")

    if args.mask:
        json_path = os.path.join(data_path, 'dataset', session, 'detections_mask.json')
        output_json_name = 'VPout_{}_r{}_mask.json'.format(model_dir_name, args.resume)
    else:
        json_path = os.path.join(data_path, 'dataset', session, 'detections.json')
        output_json_name = 'VPout_{}_r{}.json'.format(model_dir_name, args.resume)

    output_json_path = os.path.join(data_path, 'dataset', session, output_json_name)

    with open(json_path, 'r') as f:
        detection_data = json.load(f)

    total_box_count = sum([len(item['boxes']) for item in detection_data])
    print("Loaded {} bounding boxes for {} frames".format(total_box_count, len(detection_data)))
    box_cnt = 0

    start_time = time.time()

    for detection_cnt, detection in enumerate(detection_data):
        for _ in range(args.skip):
            ret, frame = cap.read()

        if not ret or frame is None:
            break

        frame_cnt_orig = cap.get(cv2.CAP_PROP_POS_FRAMES)
        frame_cnt = detection['frame_cnt']

        if frame_cnt != frame_cnt_orig:
            raise Exception("Frames from OD do not match frames now! Wrong skip param?")

        boxes = detection['boxes']
        scores = detection['scores']
        if args.mask:
            masks = detection['masks']

        if args.debug:
            show_debug(np.copy(frame), boxes)

        for i in range(len(boxes)):
            box_cnt += 1

            if args.mask:
                batch_vp_detector.process(frame, boxes[i], scores[i], frame_cnt=frame_cnt, mask=masks[i])
            else:
                batch_vp_detector.process(frame, boxes[i], scores[i], frame_cnt=frame_cnt)

        if args.dump_every != 0 and detection_cnt % args.dump_every == 0:
            print("Saving at detection ", detection_cnt)
            save(output_json_path, batch_vp_detector.output_list)

        remaining_seconds = (time.time() - start_time) / (box_cnt + 1) * (total_box_count - box_cnt)
        print('Frame {}, Box: {} / {}, ETA: {}'.format(frame_cnt, box_cnt, total_box_count, datetime.timedelta(seconds=remaining_seconds)))

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
    sessions = os.listdir(os.path.join(data_path, 'dataset'))
    for session in sessions:
        detect_session(model, model_dir_name, data_path, session, args, scales)

if __name__ == '__main__':
    detect()