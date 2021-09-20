# openvino_detector
Object Detectio using OpenVINO
By Park JaeHun

## Environments
- Ubuntu 20.04
- OpenVINO 2021.3.394
- OpenCV 4.5.2-openvino
- Numpy 1.17.3
## Requirements
- OpenVINO
- OpenCV
- Numpy

## Demo Run
1. Downloads [MOT20](https://motchallenge.net/data/MOT20/) dataset
2. Create a symbolic link to the dataset
  ```
  $ ln -s /path/to/MOT20_challenge/data/2DMOT2020 mot_benchmark
  ```
3. Downloads yolov4-tiny IR file
  ```
  $ cd /path/to/openvino_detector/IR/Yolo
  $ source ./model_downloads.sh
  $ tar -xvf yolov4-tiny_coco.tar.xz
  ```
4. Run
  ```
  $ python3 mot_detector.py --model_type=yolo --model_path=IR/Yolo/yolov4-tiny_coco  --device=CPU --display
  ```
### Note
- Detection results are automatically saved in the det folder created in origin image data path.
  
## Using in your own project
Below is the gist of how to use openvino_detector. See the ['main'](https://github.com/MilyangParkJaeHun/openvino_detector/blob/51d4ff7b63f7fa3c9bff1e746a49ca55a6d41ed3/mot_detector.py#L32) section of [mot_detector.py](https://github.com/MilyangParkJaeHun/openvino_detector/blob/51d4ff7b63f7fa3c9bff1e746a49ca55a6d41ed3/mot_detector.py) for a complete example.
```
from openvino_detector.DetModel import OpenvinoDet
from openvino_detector.Model.Yolo import Yolo
from openvino_detector.Model.Ssd import Ssd

# create model parser instance
model =  Yolo()

# create detector instance
detector = OpenvinoDet(model, "/path/to/IR", "CPU", ["person", 0, ...], 0.5)

# get frame
...

# run detection
res = detector.inference(frame)

# get MOT challenge format data
for object_dict in res:
    d = detector.to_dets(object_dict)

# get detect results image
out_frame = detector.get_results_img(res)

```
