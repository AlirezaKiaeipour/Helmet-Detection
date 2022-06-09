# Helmet Detection

**Helmet Detection Using YOLO5 Algorithm**
  
![3](https://user-images.githubusercontent.com/88143329/172671382-bcca01da-37e7-41f5-b577-c8b967321bc0.jpg)
  
  
## installation
Clone repo and install requirements.txt
  ```
  git clone https://github.com/ultralytics/yolov5  # clone
  cd Helmet-Detection
  pip install -r requirements.txt  # install
  ```
  
## Download Dataset

Download the dataset from this **[link](https://www.kaggle.com/datasets/alirezakiaipoor/helmet)**

## Create dataset.yaml

  ```
path: ./helmet  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/valid  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes
nc: 80  # number of classes
names: [ 'head', 'helmet']  # class names
  ```
  
  
## Select a Model
Select a pretrained model to start training from. Here we select ``YOLOv5s``, the smallest and fastest model available.

<br />

![2](https://user-images.githubusercontent.com/88143329/172703629-007d9320-3b4f-4c61-a874-7b438d894da2.png)



|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>CPU b1<br>(ms) |Speed<br><sup>V100 b1<br>(ms) |Speed<br><sup>V100 b32<br>(ms) |params<br><sup>(M) |FLOPs<br><sup>@640 (B)
|---                    |---  |---    |---    |---    |---    |---    |---    |---
|YOLOv5n     |640  |28.0   |45.7   |**45** |**6.3**|**0.6**|**1.9**|**4.5**
|YOLOv5s      |640  |37.4   |56.8   |98     |6.4    |0.9    |7.2    |16.5
|YOLOv5m      |640  |45.4   |64.1   |224    |8.2    |1.7    |21.2   |49.0
|YOLOv5l      |640  |49.0   |67.3   |430    |10.1   |2.7    |46.5   |109.1
|YOLOv5x      |640  |50.7   |68.9   |766    |12.1   |4.8    |86.7   |205.7
|                       |     |       |       |       |       |       |       |
|YOLOv5n6     |1280 |36.0   |54.4   |153    |8.1    |2.1    |3.2    |4.6
|YOLOv5s6     |1280 |44.8   |63.7   |385    |8.2    |3.6    |12.6   |16.8
|YOLOv5m6     |1280 |51.3   |69.3   |887    |11.1   |6.8    |35.7   |50.0
|YOLOv5l6     |1280 |53.7   |71.3   |1784   |15.8   |10.5   |76.8   |111.4

  
## Train
  
``Train`` a pretrained model on helmet dataset
  
```
# Train YOLOv5s on helmet for 300 epochs
$ python train.py --img 640 --batch 16 --epochs 300 --data dataset.yaml --weights yolov5s.pt
```  
## Test
Run the following command for evaluation trained model on ``test`` dataset:
```
$ python val.py --data dataset.yaml --weights ./weights/best.pt --img 640 --batch 1
```  
  
## Inference
Run the following command for ``Inference``
```
$ python detect.py --weights ./weights/best.pt --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
``` 
  
## Export a Trained Model
  - TensorRT may be up to 2-5X faster than PyTorch on GPU
  - ONNX and OpenVINO may be up to 2-3X faster than PyTorch on CPU
```
$ python export.py --weights ./weights/best.pt --device [CPU:cpu , GPU:0] --include
                          torchscript  # TorchScript
                          onnx  # ONNX
                          openvino  # OpenVINO
                          engine  # TensorRT
                          coreml  # CoreML
                          saved_model  # TensorFlow SavedModel
                          pb  # TensorFlow GraphDef
                          tflite  # TensorFlow Lite
                          edgetpu  # TensorFlow Edge TPU
                          tfjs  # TensorFlow.js
``` 

## Inference Time
Compare inference time of exported models
  
  Model | Pytorch<br>(ms) | TensorRT<br>(ms) | ONNX<br>(ms) | TensorFlow Lite<br>(ms) |
  ------------- | ------------- | ------------- | ------------- | ------------- |
  ``best.pt`` | **11.9** | **7.1** | **12.2**  | **426.5** |
  
## Reference
  
https://github.com/ultralytics/yolov5
