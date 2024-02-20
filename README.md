
# SIINN
Simple Inspecting and Inferencing Neural Network (SIINN) is an open source tool that empowers AI developers to fastly access their models. Quickly inspect the input&output information of the model, and run a speed test program to exam the throughput of the model.

SINN is widely supported in many deployed neural network models.

## Install SINN
```
pip install git+https://github.com/chenxinfeng4/siinn.git
siinn inspect ./yolovn8.onnx|om|xml|engine|rknn
siinn run ./yolovn8.onnx
```

## Supported neural network platforms.
- ONNX: *.onnx
- Tensorrt: *.engine
- OpenVINO: *.xml
- RKNN (rockchip rk3588): *.rknn
- HiAscend (mindx): *.om
- CoreML: *.mlpackage
- (on the way) TensorFlow: *.pd

## Inspect the input&output layers
The input&output layers is a interface of the model file. It's important to get the name, shape and dtype of the input&output layers before you can use a model.

```
# Select your existed file: yolov8n.onnx *.engine *.rknn *.om *.xml *.pd *.mlmodel
siinn inspect ./yolovn8.onnx
```
You get the information of the model. There is one input, input name is `images`, data type in `float32`, data shape is `(1, 3, 640, 640)`. The output layer names `output0`. If the model contain dynamic shape, than the dynamic axis would be represent as '-1' or '0'.

```
[I] Loading model: yolov8n.onnx
[I] ==== ONNX Model ====
    Name: torch-jit-export | ONNX Opset: 10
    
    ---- 1 Graph Input(s) ----
    images [dtype=float32, shape=(1, 3, 640, 640)]
    
    ---- 1 Graph Output(s) ----
    output0 [dtype=float32, shape=(1, 84, 8400)]
    
    ---- 134 Initializer(s) ----
    
    ---- 237 Node(s) ----
```

You can inspect other model files in similar way. For example `siinn inspect ./yolovn8.engine` for NVIDIA TensorRT model. The OpenVINO, RKNN, HiAscend are also supported.

> Make sure you have already installed the SDK in your focus platform. For example, if you want the onnx, please manully `pip install onnx`. If you want the Tensorrt, please install the TensorRT libs and python packages. The SIINN would not download and install these environment.

## Run the model and see the inference speed.
You get the speed of the model by executing this command. It will feed the input layers from the registered data shape and type. The program is pure model inferencing, didn't cover image preprocessing and postprocessing.

```
# Select your existed file: yolov8n.onnx *.engine *.rknn *.om *.xml *.pd *.mlmodel
siinn run ./yolovn8.onnx
```
The `(1, 3, 640, 640) float32` data is automatically assigned to the input layer by inspecting the model information. The throughtput is `47.62` fps for yolov8n in onnx model.

```
input  (1, 3, 640, 640) float32
output (1, 84, 8400) float32
6%|█████                                     | 120/2000 [00:02<00:39, 47.62it/s]
```

You can run other model files in similar way. For example `siinn run ./yolovn8.engine`.
