"""
    DetModel.py

    Author: Park Jaehun
    Refactoring: Park Jaehun , 2021.09.18
"""

from numpy.core.numeric import outer
from .Model.ModelParser import ModelParser
from abc import *

import os
import cv2
import sys
import glob
import numpy as np
from openvino.inference_engine import IECore

class OpenvinoDet():
    def __init__(self, model_parser: ModelParser, model_path, device, label_map, prob_threshold=0.5):
        """
        Set key parameters for Detector
        """
        self.before_frame = None # for frame synchronization
        self.current_frame = None
        self.model_path = model_path
        self.device = device
        self.label_map = label_map
        self.prob_threshold = prob_threshold
        self.iou_threshold = 0.4

        self.img_height = 480
        self.img_width  = 640
        self.before_frame = np.empty((self.img_height, self.img_width, 3))

        self.model_bin = glob.glob(os.path.join(self.model_path, '*.bin'))
        self.model_xml = glob.glob(os.path.join(self.model_path, '*.xml'))
        if not self.model_bin or not self.model_xml:
            print("can not find IR model")
            sys.exit(1)
        else:
            self.model_bin = self.model_bin[0]
            self.model_xml = self.model_xml[0]

        self._model = model_parser

        self.num_requests = 2
        self.cur_request_id = 0
        self.next_request_id = 1

        print(self.model_path)
        print("Reading IR...")
        self.ie = IECore()
        self.net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)

        input_name = list(self.net.input_info.keys())[0]
        self.input_blob = self.net.input_info[input_name].input_data.name
        self.out_blob = next(iter(self.net.outputs))
        self.n, self.c, self.h, self.w = self.net.input_info[input_name].input_data.shape

        print("Loading IR to the plugin...")
        self.exec_net = self.ie.load_network(
            network=self.net, device_name=self.device, num_requests=self.num_requests)
        print("Successfully load")

    def inference(self, frame):
        """
        Inference of the detection model with frame as input.
        
        Because the Object Detection Model works asynchronously, 
        the current result is the detection result of the previous frame
        
        output : [{xmin, ymin, xmax, ymax, class_id, confidence}, ... ]
        """
        self.img_height = frame.shape[0]
        self.img_width = frame.shape[1]
        in_frame = self.preprocess_frame(frame)

        self.exec_net.start_async(request_id=self.next_request_id, inputs={self.input_blob: in_frame})
        dets = list()
        if self.exec_net.requests[self.cur_request_id].wait(-1) == 0:
            output = self._model.get_output(self)
            dets = self._model.parse_output(self, output)
            dets = self.filter_dets(dets)
            dets = self.cut_outer_dets(dets)
        self.switching_request_id()
        self.update_frame(frame)
        return dets

    def clear(self):
        """
        Reload Object Detecton Model to refresh detection results buffer
        """
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=self.num_requests)
        print("Successfully reload!!!")

    def update_frame(self, frame):
        """
        Update before frame to current frame
        """
        self.before_frame = self.current_frame
        self.current_frame = frame.copy()

    def get_results_img(self, dets):
        """
        Draw the infrerence results from Object Detection Model on frame

        Since object detection is performed asynchronously, 
        the current detection result is drawn in the previous frame.

        output : frame
        """
        if hasattr(self.before_frame, 'size'):
            for det in dets:
                color = (100, 255, 100)
                det_label = self.label_map[det['class_id']] if self.label_map and len(self.label_map) >= det['class_id'] \
                            else str(det['class_id'])

                cv2.rectangle(self.before_frame, (det['xmin'], det['ymin']), (det['xmax'], det['ymax']), color, 7)
                cv2.putText(self.before_frame,
                            str(round(det['confidence'] * 100, 1)) + ' %',
                            (det['xmin'], det['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)  
        else:
            return self.current_frame

        return self.before_frame

    def intersection_over_union(self, box_1, box_2):#add DIOU-NMS support
        """
        Calculate IOU score between two bounding boxes
        """
        width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
        height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])

        cw = max(box_1['xmax'], box_2['xmax'])-min(box_1['xmin'], box_2['xmin'])
        ch = max(box_1['ymax'], box_2['ymax'])-min(box_1['ymin'], box_2['ymin'])
        c_area = cw**2+ch**2+1e-16
        rh02 = ((box_2['xmax']+box_2['xmin'])-(box_1['xmax']+box_1['xmin']))**2/4+((box_2['ymax']+box_2['ymin'])-(box_1['ymax']+box_1['ymin']))**2/4

        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
        box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
        area_of_union = box_1_area + box_2_area - area_of_overlap
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union-pow(rh02/c_area,0.6)

    def preprocess_frame(self, frame):
        """
        Preprocess frame according to input format of detector model.
        """
        in_frame = cv2.resize(frame, (self.w, self.h),
                            interpolation=cv2.INTER_CUBIC)
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))

        return in_frame
    
    def filter_dets(self, dets):
        """
        Filter only values ​​to be used in detection results.
        1. Apply NMS(Non Maximum Suppression).
        2. Apply probability threshold.
        3. Apply class id filtering. (only detect person)
        """
        dets = sorted(dets, key=lambda det : det['confidence'], reverse=True)
        for i in range(len(dets)):
            if dets[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(dets)):
                if self.intersection_over_union(dets[i], dets[j]) > self.iou_threshold:
                    dets[j]['confidence'] = 0
        return tuple(det for det in dets if (det['confidence'] >= self.prob_threshold))

    def cut_outer_dets(self, dets):
        """
        Prevent bounding box from exceeding frame size.
        """
        for det in dets:
            det['xmax'] = min(det['xmax'], self.img_width)
            det['ymax'] = min(det['ymax'], self.img_height)
            det['xmin'] = max(det['xmin'], 0)
            det['ymin'] = max(det['ymin'], 0)

        return dets
    
    def switching_request_id(self):
        """
        Switch id for async detection mode.
        """
        self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id

    def to_dets(self, dict):
        """
        Convert the detection results format 

        from dict format to dets format used for MOTChallengeEvalKit

        dict : [class_id, xmin, ymin, xmax, ymax, confidence]
        dets : [frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z]
        """
        dets = {}
        dets['bb_left']     = dict['xmin']
        dets['bb_top']      = dict['ymin']
        dets['bb_width']    = dict['xmax'] - dict['xmin']
        dets['bb_height']   = dict['ymax'] - dict['ymin']
        dets['conf']        = dict['confidence']

        return dets