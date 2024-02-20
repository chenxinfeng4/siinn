#!/bin/env python

import argparse
import os.path as osp
from typing import List

def tprint(x, prefix='   '):
    print(prefix, x)


def get_node_info(node):
    import onnx
    node_name = node.name
    node_shape = node.type.tensor_type.shape.dim
    node_size = tuple([(dim.dim_value if dim.dim_value>0 else dim.dim_param)
                       for dim in node_shape])

    node_dtype = node.type.tensor_type.elem_type
    dtype_str = onnx.TensorProto.DataType.Name(node_dtype).lower()
    if dtype_str == "float":
        dtype_str = "float32"
    return node_name, dtype_str, node_size


def disp_inputs_outputs(input_layers:List[dict], output_layers:List[dict]):
    num_inputs = len(input_layers)
    tprint(f"---- {num_inputs} Graph Input(s) ----")
    for layer in input_layers:
        idx = f'#{layer["idx"]}' if 'idx' in layer else ''
        name = layer.get('name', '')
        shape = tuple(layer['shape'])
        dtype = layer['dtype']
        idx_name = ' '.join([idx, name]).strip()
        tprint(f'{idx_name} [dtype={dtype}, shape={shape}]')
    tprint('')

    num_outputs = len(output_layers)
    tprint(f"---- {num_outputs} Graph Output(s) ----")
    for layer in output_layers:
        idx = f'#{layer["idx"]}' if 'idx' in layer else ''
        name = layer.get('name', '')
        shape = tuple(layer['shape'])
        dtype = layer['dtype']
        idx_name = ' '.join([idx, name]).strip()
        tprint(f'{idx_name} [dtype={dtype}, shape={shape}]')


def inspect_onnx(onnx_file) -> list:
    import onnx
    # Load the ONNX model
    model = onnx.load(onnx_file)

    # print headers
    tprint(f'Loading model: {onnx_file}', prefix='[I]')
    tprint('==== ONNX Model ====', prefix='[I]')
    model_name = model.graph.name
    opset_version = None
    for opset in model.opset_import:
        if opset.domain == '':
            opset_version = opset.version
            break
    tprint(f'Name: {model_name} | ONNX Opset: {opset_version}')
    tprint('')

    input_layers, output_layers = [], []
    for node in model.graph.input:
        name, dtype_str, shape = get_node_info(node)
        input_layers.append({'name':name, 'shape':shape, 'dtype':dtype_str})

    for node in model.graph.output:
        name, dtype_str, shape = get_node_info(node)
        output_layers.append({'name':name, 'shape':shape, 'dtype':dtype_str})

    disp_inputs_outputs(input_layers, output_layers)

    # print Other layers
    num_initializers = len(model.graph.initializer)
    num_nodes = len(model.graph.node)
    tprint('')
    tprint(f"---- {num_initializers} Initializer(s) ----")
    tprint('')
    tprint(f"---- {num_nodes} Node(s) ----")
    return model, input_layers, output_layers


def inspect_om(model_path) -> list:
    from mindx.sdk import base
    model = base.model(modelPath=model_path, deviceId=0)  # 初始化 base.model 类

    # print headers
    tprint(f'Loading model: {model_path}', prefix='[I]')
    tprint('==== OM Model ====', prefix='[I]')
    tprint('')

    # print INPUTs
    num_inputs = model.input_num
    input_layers, output_layers = [], []
    for i in range(num_inputs):
        item = dict()
        item['shape'] = tuple(model.input_shape(i))
        item['idx'] = f'#{i}'
        item['dtype'] = model.input_dtype(i).name
        input_layers.append(item)

    # print OUTPUTs
    num_outputs = model.output_num
    tprint(f"---- {num_outputs} Graph Output(s) ----")
    for i in range(num_outputs):
        item = dict()
        item['shape'] = tuple(model.output_shape(i))
        item['idx'] = f'#{i}'
        item['dtype'] = model.output_dtype(i).name
        output_layers.append(item)
    disp_inputs_outputs(input_layers, output_layers)
    return model, input_layers, output_layers


def inspect_trt(model_path) -> list:
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INTERNAL_ERROR)

    def dtype_to_str(dtype):
        dtype_str = dtype.name.lower()
        if dtype_str == "float":
            dtype_str = "float32"
        return dtype_str

    with trt.Runtime(logger) as runtime:
        with open(model_path, 'rb') as f, runtime.deserialize_cuda_engine(f.read()) as engine:
            # 获取输入张量的信息
            num_bindings = engine.num_bindings
            names = [engine.get_tensor_name(i) for i in range(num_bindings)]
            isinputs = [engine.get_tensor_mode(n).name == 'INPUT' for n in names]
            shapes = [engine.get_tensor_shape(n) for n in names]
            dtypes = [dtype_to_str(engine.get_tensor_dtype(n)) for n in names]
            
    tprint(f'Loading model: {model_path}', prefix='[I]')
    tprint('==== Tensorrt Model ====', prefix='[I]')
    tprint('')

    input_layers, output_layers = [], []
    for i, (isinput, name, shape, dtype) in enumerate(zip(isinputs, names, shapes, dtypes)):
        item = {'idx':i, 'name':name, 'shape':shape, 'dtype':dtype}
        if isinput:
            input_layers.append(item)
        else:
            output_layers.append(item)

    disp_inputs_outputs(input_layers, output_layers)
    return None, input_layers, output_layers


def inspect_openvino(model_path:str):
    from openvino.runtime import Core
    ie=Core()
    model = ie.read_model(model=model_path)
    input_layers, output_layers = [], []
    for layer in model.inputs:
        item = dict(name = next(iter(layer.names)),
                    dtype = layer.element_type.to_dtype().name,
                    shape = tuple(layer.shape))
        input_layers.append(item)

    for layer in model.outputs:
        item = dict(name = next(iter(layer.names)),
                    dtype = layer.element_type.to_dtype().name,
                    shape = tuple(layer.shape))
        output_layers.append(item)
    
    tprint(f'Loading model: {model_path}', prefix='[I]')
    tprint('==== OpenVINO Model ====', prefix='[I]')
    tprint('')
    disp_inputs_outputs(input_layers, output_layers)
    return model, input_layers, output_layers


def inspect_coreml(model_path:str) -> list:
    import coremltools as ct
    model = ct.models.MLModel(model_path)
    description = model._spec.description
    input_layers, output_layers = [], []
    for layer in description.input:
        name = layer.name
        array = layer.type.multiArrayType
        shape = tuple(array.shape)
        dtype = str(array).strip().split(" ")[-1].lower()
        item = {"name" : name, "shape" : shape, "dtype" : dtype}
        input_layers.append(item)

    for layer in description.output:
        name = layer.name
        array = layer.type.multiArrayType
        shape = tuple(array.shape)
        dtype = str(array).strip().split(" ")[-1].lower()
        item = {"name" : name, "shape" : shape, "dtype" : dtype}
        output_layers.append(item)

    tprint(f'Loading model: {model_path}', prefix='[I]')
    tprint('==== CoreML Model ====', prefix='[I]')
    tprint('')
    disp_inputs_outputs(input_layers, output_layers)
    return model, input_layers, output_layers

def inspect_keras(model_path:str) -> list:
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    model.input.shape
    model.input.dtype

def inspect_proxy(model_path:str, quiet=False) -> list:
    assert osp.isfile(model_path) or osp.isdir(model_path)
    ext = osp.splitext(model_path)[-1]
    global tprint
    _tprint = tprint
    if quiet:
        tprint = lambda *args, **kwargs: None

    if ext == '.onnx':
        inspect_onnx(model_path)
    elif ext == '.om':
        inspect_om(model_path)
    elif ext in ['.engine', '.trt']:
        inspect_trt(model_path)
    elif ext == '.xml':
        inspect_openvino(model_path)
    elif ext in ['.mlpackage', '.mlpackage/']:
        inspect_coreml(model_path)
    else:
        tprint = _tprint
        raise ValueError(f'Unsupported model extension: {ext}')
    tprint = _tprint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX Model Info Printer')
    parser.add_argument('model_path', type=str, help='Path to ONNX model')
    args = parser.parse_args()
    model_path = args.model_path
    inspect_proxy(args.model_path)
