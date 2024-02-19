#!/bin/env python
#echo "0 1 2 3 4 5" | xargs -P 0 -n 1 onnx_speed /home/chenxinfeng/ml-project/rknn_speedtest/resnet50.rknn --gpu
import numpy as np
import tqdm
import argparse
import os.path as osp


def get_node_info(node):
    import onnx
    node_name = node.name
    node_shape = node.type.tensor_type.shape.dim
    node_size = tuple([max(1, dim.dim_value) for dim in node_shape])

    node_dtype = node.type.tensor_type.elem_type
    dtype_str = onnx.TensorProto.DataType.Name(node_dtype).lower()
    if dtype_str == "float":
        dtype_str = "float32"
    return node_name, dtype_str, node_size


def main_onnx(onnx_file:str):
    import onnx
    import onnxruntime
    sess = onnxruntime.InferenceSession(onnx_file)

    # Load the ONNX model
    model = onnx.load(onnx_file)
    num_inputs = len(model.graph.input)

    # print INPUTs
    IN_dict = dict()
    for i in range(num_inputs):
        node_name, dtype_str, node_size = get_node_info(model.graph.input[i])
        IN_dict[node_name] = np.random.random(node_size).astype(dtype_str)

    result = sess.run(None, IN_dict)
    
    for k,v in IN_dict.items():
        print('input', k, v.shape, v.dtype)
    for r in result:
        print('output', r.shape, r.dtype)

    for _ in tqdm.trange(2000):
        result = sess.run(None, IN_dict)


def main_om(model_path:str, igpu:int):
    from mindx.sdk import base, Tensor
    model = base.model(modelPath=model_path, deviceId=0)  # 初始化 base.model 类

    # print INPUTs
    num_inputs = model.input_num
    IN_list = []
    for i in range(num_inputs):
        node_size = tuple(model.input_shape(i))
        dtype_str = model.input_dtype(i).name
        IN_list.append(np.random.random(node_size).astype(dtype_str))

    #
    '''模型推理'''
    IN_tensors = []
    for img in IN_list:
        img = Tensor(img)
        img.to_device(0)
        IN_tensors.append(img)
    
    result = model.infer(IN_tensors)  # 执行推理。输入数据类型：List[base.Tensor]， 返回模型推理输出的 List[base.Tensor]
    for v in IN_list:
        print('input', v.shape, v.dtype)
    for r in result:
        print('output', tuple(r.shape), r.dtype.name)
    
    for _ in tqdm.trange(2000, position=igpu):
        result = model.infer(IN_tensors)


def main_rknn(model_path:str, igpu:int):
    # Get input shape
    import onnx
    onnx_file = model_path.replace('.rknn', '.onnx')
    assert osp.isfile(onnx_file), 'Sibling onnx file not exist.'
    model = onnx.load(onnx_file)
    num_inputs = len(model.graph.input)
    IN_list = []
    for i in range(num_inputs):
        _, _, node_size = get_node_info(model.graph.input[i])
        IN_list.append(np.random.randint(0,255, node_size).astype(np.uint8))
    
    # Load RKNN model
    from rknnlite.api import RKNNLite
    rknn_lite = RKNNLite(verbose=False)
    print(f'--> Load RKNN model {model_path}')
    ret = rknn_lite.load_rknn(model_path)
    assert ret == 0, 'Load RKNN model failed'

    # Define gpu        
    core_mask = 2**(igpu%3)
    ret = rknn_lite.init_runtime(core_mask=core_mask) #core_mask=RKNNLite.NsPU_CORE_0

    # Inference
    result = rknn_lite.inference(inputs=IN_list)
    for v in IN_list:
        print('input', v.shape, v.dtype)
    for r in result:
        print('output', r.shape, r.dtype)

    for _ in tqdm.trange(5000, position=igpu):
        result = rknn_lite.inference(inputs=IN_list)


def main_trt(model_path:str, igpu:int):
    import torch
    try:
        from torch2trt import TRTModule
        from torch2trt.torch2trt import torch_dtype_from_trt
    except:
        print('pip install git+https://github.com/chenxinfeng4/torch2trt')
        raise Exception('Not install!')
    
    num_gpus = torch.cuda.device_count()
    igpu_ = igpu % num_gpus
    with torch.cuda.device(f'cuda:{igpu_}'):
        trt_model = TRTModule()
        trt_model.load_from_engine(model_path)

        IN_list = []
        for i in range(len(trt_model.input_names)):
            name = trt_model.input_names[i]
            input_dtype = torch_dtype_from_trt(trt_model.engine.get_tensor_dtype(name))
            input_shape = np.array(trt_model.context.get_tensor_shape(name))
            input_shape[input_shape<=0] = 1
            img_NCHW = np.zeros(input_shape)
            batch_img = torch.from_numpy(img_NCHW).cuda().type(input_dtype)
            IN_list.append(batch_img)

        result = trt_model(*IN_list)
        if type(result) not in (list,tuple):
            result = [result]
        
        for name, v in zip(trt_model.input_names, IN_list):
            v = v.cpu().numpy()
            print('input', name, v.shape, v.dtype)
        for name, r in zip(trt_model.output_names, result):
            r = r.cpu().numpy()
            print('output', name, r.shape, r.dtype)
        
        for _ in tqdm.trange(50000, position=igpu):
            result = trt_model(*IN_list)
            torch.cuda.current_stream().synchronize()


def create_IN_list_np(input_layers):
    out_list = [np.random.random(l['shape']).astype(l['dtype'])
                for l in input_layers]
    return out_list


def main_openvino(model_path:str):
    from openvino.runtime import Core
    ie=Core()
    net = ie.read_model(model=model_path)
    model = ie.compile_model(model=net, device_name='CPU')
    
    input_layers = []
    for layer in model.inputs:
        item = dict(name = next(iter(layer.names)),
                    dtype = layer.element_type.to_dtype().name,
                    shape = tuple(layer.shape))
        input_layers.append(item)
    IN_list = create_IN_list_np(input_layers)
    result = model(IN_list)
    for l, v in zip(input_layers, IN_list):
        print('input', l['name'], v.shape, v.dtype)
    for l, r in result.items():
        print('output', next(iter(l.names)), r.shape, r.dtype)
    
    for _ in tqdm.trange(2000):
        result = model(IN_list)


def main_inference(model_path:str, igpu:int):
    assert osp.isfile(model_path)
    ext = osp.splitext(model_path)[-1]

    if ext == '.onnx':
        main_onnx(model_path)
    elif ext == '.om':
        main_om(model_path, igpu)
    elif ext == '.rknn':
        main_rknn(model_path, igpu)
    elif ext in ['.engine', '.trt']:
        main_trt(model_path, igpu)
    elif ext in ['.xml']:
        main_openvino(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX Model Info Printer')
    parser.add_argument('model_path', type=str, help='Path to ONNX model')
    parser.add_argument('-i', '--ithread', type=int, default=0, help='Index of thread/gpu')
    args = parser.parse_args()
    model_path = args.model_path
    igpu = args.ithread
    main_inference(model_path, igpu)
