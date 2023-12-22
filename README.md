# DRIVER MONITORING SYSTEM USING JETSON NANO AND SSD MOBILE NET.
![JetsonNano](https://github.com/Mo-Alsehli/Driver_Monitoring_System_JetsonNano_SSDMobileNet/assets/98949843/02a20108-aa39-4901-85c5-844339a784a1)
![image](https://github.com/Mo-Alsehli/Driver_Monitoring_System_JetsonNano_SSDMobileNet/assets/98949843/5d12b0bc-5f8f-4e49-9622-d63ed4166a80)

# Summery
- This is an SSD Mobile Net pre-trained model that is customized for a driver monitoring system.
- Main Features:
  - Detect Open Eyes and Closed Eyes.
  - Detect Detect Drowsniss.
  - Detect Phone.
 
## Technologies:
- Nvidia Jetson Nano Board.
- Jetpack: 4.6
- Tensorflow.
- Tensorflow Object Detection API.
- SSD Mobile Net Pre-trained model.
- TensorRT.
- Other Libraries (numpy, ...etc).

## Procedure:
1. We downloaded our model and then customized it with our labels.
2. The Full Description For The Model Is Here: [DMS Model](https://github.com/Mo-Alsehli/Driver_Monitoring_System_JetsonNano_SSDMobileNet/tree/master/SSD_MobileNet_Model).
3. Create FreezGraph:
```
python exporter_main_v2.py \
    --input_type float_image_tensor \
    --trained_checkpoint_dir /path/to/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint \
    --pipeline_config_path /path/to/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config \
    --output_directory /path/to/export
```
4. Create Onnx:
- The ONNX interchange format provides a way to export models from many frameworks, including PyTorch, TensorFlow, and TensorFlow 2, for use with the TensorRT runtime.
- Here is the command to create your Onnx:
```
python create_onnx.py \
    --pipeline_config /path/to/exported/pipeline.config \
    --saved_model /path/to/exported/saved_model \
    --onnx /path/to/save/model.onnx
```
5. Create TRT Engine:
- NVIDIA® TensorRT™ is an SDK for optimizing trained deep learning models to enable high-performance inference.
- TensorRT contains a deep learning inference optimizer for trained deep learning models, and a runtime for execution.
- After you have trained your deep learning model in a framework of your choice, TensorRT enables you to run it with higher throughput and lower latency.
![image](https://github.com/Mo-Alsehli/Driver_Monitoring_System_JetsonNano_SSDMobileNet/assets/98949843/fdd9236d-719b-4bfc-b2f4-8b06682f846f)
- Create TRT Engine:
  - NOTE-> Where is TensorRT: `/usr/src/tensorrt/bin`
```
trtexec --onnx=resnet50_onnx_model.onnx --saveEngine=engine.trt
```

# Results:
- Finally our Trt Engine Works on average 20FPS.

### Resources:
- [Resource-1](https://github.com/NVIDIA/TensorRT/tree/release/8.2/samples/python/tensorflow_object_detection_api).
- [Resource-2](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#export-from-tf).
- [Resource-3](https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/3.%20Using%20Tensorflow%202%20through%20ONNX.ipynb).
- [Resource-4](https://www.youtube.com/watch?v=yqkISICHH-U&t=16912s).
- [Resource-5](https://github.com/nicknochnack/TFODCourse).


