"""
    Ssd.py

    Author: Park Jaehun
    Refactoring: Park Jaehun , 2021.09.18
"""

from .ModelParser import ModelParser

class Ssd(ModelParser):
    def get_output(self, model):

        return model.exec_net.requests[model.cur_request_id].output_blobs

    def parse_output(self, model, output):
        src_h, src_w = model.img_height, model.img_width
        dets = list()

        output_name = list(output.keys())[0]
        output = output[output_name].buffer.flatten()
        total_len = int(len(output)/7)

        for idx in range(total_len):
            base_index = idx * 7
            class_id = int(output[base_index + 1]) - 1
            prob = float(output[base_index + 2])
            xmin = int(output[base_index + 3] * src_w)
            ymin = int(output[base_index + 4] * src_h)
            xmax = int(output[base_index + 5] * src_w)
            ymax = int(output[base_index + 6] * src_h)

            det = dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=prob)
            dets.append(det)

        return dets