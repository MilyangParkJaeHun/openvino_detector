"""
    mot_detector.py

    Author: Park Jaehun
    Refactoring: Park Jaehun , 2021.09.19
"""

import os
import cv2
import sys
import glob
import argparse
import numpy as np
import time
from DetModel import OpenvinoDet
from Model.Yolo import Yolo
from Model.Ssd import Ssd
from Model.PedestrianDet import PedestrianDet

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object Detection demo')
    parser.add_argument('--model_type', help='Type of object detection model : [ssd / yolo / ped]', type=str, default='')
    parser.add_argument('--model_path', help='Path to object detection model weight file', type=str, default='IR/Yolo/coco')
    parser.add_argument('--img_path', help='Path to input images.', type=str, default='mot_benchmark')
    parser.add_argument('--device', help='Device for inference', type=str, default='GPU')
    parser.add_argument("--prob_threshold", help='Minimum probability for detection.', type=float, default=0.5)
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]', action='store_true')
    parser.add_argument('--phase', help="Subdirectory in seq_path.", type=str, default='train')
    args = parser.parse_args()
    return args   

if __name__ == "__main__":
    args = parse_args()

    display = args.display
    phase = args.phase

    # for calculate FPS
    total_time = 0 
    total_frame = 0

    # only detect person
    label_map = [''] * 90
    label_map[0] = 'person'

    # create model parser according to model type[yolo/ssd]
    if args.model_type == "yolo":
        model = Yolo()
    elif args.model_type == "ssd":
        model = Ssd()
    elif args.model_type == 'ped':
        model = PedestrianDet()
    else:
        print("Supports ssd or yolo or ped as detection models. \n Choose between ssd / yolo / ped!!!")
        sys.exit(1)

    # create Detector instance
    detector = OpenvinoDet(model, args.model_path, args.device, label_map, args.prob_threshold)

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.img_path, phase, '*', 'img1')

    for img_dir in glob.glob(pattern):
        # clear detection results buffer 
        detector.clear()

        seq = img_dir[pattern.find('*'):].split(os.path.sep)[0]
        out_det_path = os.path.join(args.img_path, phase, seq, 'det')
        if not os.path.exists(out_det_path):
            os.makedirs(out_det_path)

        out_img_dir = os.path.join('output', seq, 'img')
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)

        with open(os.path.join(out_det_path, 'det.txt'), 'w') as out_file:
            # for using last frame, add one more time last frame.
            img_id_list = [i+1 for i in range(len(os.listdir(img_dir)))]
            img_id_list.append(img_id_list[-1])

            frame_id = 0
            for img_id in img_id_list:
                frame_id += 1
                img_file = '%06d.jpg'%(img_id)

                img_file_path = os.path.join(img_dir, img_file)
                frame = cv2.imread(img_file_path, cv2.IMREAD_COLOR)

                # inference
                det_start_time = time.time()
                res = detector.inference(frame)
                det_end_time = time.time()

                total_time += det_end_time - det_start_time
                total_frame += 1
                
                before_frame_id = frame_id - 1
                for object_dict in res:
                    d = detector.to_dets(object_dict)

                    print('%d,-1,%d,%d,%d,%d,1,-1,-1,-1'\
                            %(before_frame_id, d['bb_left'], d['bb_top'], d['bb_width'], d['bb_height']),file=out_file)

                if display:
                    out_img_path = os.path.join(out_img_dir, '%06d.jpg'%(before_frame_id))
                    out_frame = detector.get_results_img(res)

                    cv2.imshow('detector', out_frame)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        sys.exit(0)
                    
                    cv2.imwrite(out_img_path, out_frame)
    
    if total_time == 0:
        raise Exception('No input received')
    print('fps : ', total_frame / total_time)
                    




