"""
    Yolo.py

    Author: Park Jaehun
    Refactoring: Park Jaehun , 2021.09.18
"""

from ModelParser import ModelParser
from math import exp as exp
import numpy as np
import ngraph as ng

class Yolo(ModelParser):
    def get_output(self, model):
        return model.exec_net.requests[model.cur_request_id].output_blobs

    def parse_output(self, model, output):
        new_frame_height_width = (model.h, model.w)
        source_height_width = (model.img_height, model.img_width)
        dets = list()
        function = ng.function_from_cnn(model.net)
        for layer_name, out_blob in output.items():
            out_blob = out_blob.buffer.reshape(model.net.outputs[layer_name].shape)
            params = [x.get_attributes() for x in function.get_ordered_ops() if x.get_friendly_name() == layer_name][0]
            layer_params = YoloParams(params, out_blob.shape[2])
            dets += self.parse_yolo_region(out_blob, new_frame_height_width, source_height_width, layer_params,
                                        model.prob_threshold)
        return dets

    def scale_bbox(self, x, y, height, width, class_id, confidence, im_h, im_w):
        xmin = int((x - width / 2) * im_w)
        ymin = int((y - height / 2) * im_h)
        xmax = int(xmin + width * im_w)
        ymax = int(ymin + height * im_h)
        # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
        # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
        return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())

    def parse_yolo_region(self, predictions, resized_image_shape, original_im_shape, params, threshold):
        # ------------------------------------------ Validating output parameters ------------------------------------------
        _, _, out_blob_h, out_blob_w = predictions.shape
        assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                        "be equal to width. Current height = {}, current width = {}" \
                                        "".format(out_blob_h, out_blob_w)

        # ------------------------------------------ Extracting layer parameters -------------------------------------------
        orig_im_h, orig_im_w = original_im_shape
        resized_image_h, resized_image_w = resized_image_shape
        dets = list()
        size_normalizer = (resized_image_w, resized_image_h) if params.isYoloV3 else (params.side, params.side)
        bbox_size = params.coords + 1 + params.classes
        # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
        for row, col, n in np.ndindex(params.side, params.side, params.num):
            # Getting raw values for each detection bounding box
            bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
            x, y, width, height, object_probability = bbox[:5]
            class_probabilities = bbox[5:]
            if object_probability < threshold:
                continue
            # Process raw value
            x = (col + x) / params.side
            y = (row + y) / params.side
            # Value for exp is very big number in some cases so following construction is using here
            try:
                width = exp(width)
                height = exp(height)
            except OverflowError:
                continue
            # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
            width = width * params.anchors[2 * n] / size_normalizer[0]
            height = height * params.anchors[2 * n + 1] / size_normalizer[1]

            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]*object_probability
            if confidence < threshold:
                continue
            dets.append(self.scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence, \
                                        im_h=orig_im_h, im_w=orig_im_w))
        return dets

class YoloParams:
  # ------------------------------------------- Extracting layer parameters ------------------------------------------
  # Magic numbers are copied from yolo samples
  def __init__(self, param, side):
    self.num = 3 if 'num' not in param else int(param['num'])
    self.coords = 4 if 'coords' not in param else int(param['coords'])
    self.classes = 80 if 'classes' not in param else int(param['classes'])
    self.side = side
    self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0,
                        373.0, 326.0] if 'anchors' not in param else param['anchors']

    self.isYoloV3 = False

    if param.get('mask'):
        mask = param['mask']
        self.num = len(mask)

        maskedAnchors = []
        for idx in mask:
            maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
        self.anchors = maskedAnchors

        self.isYoloV3 = True
