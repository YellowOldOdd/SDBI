# -*- coding: utf-8 -*-
"""

onnx to tensorRT convert tool

@author: muxiao
"""
import os.path
import tensorrt as trt
import pycuda.driver as cuda


def build_engine(onnx_file_path,
                 engine_save_path,
                 use_fp16=True,
                 use_dynamic_batch=True,
                 max_batch_size=1,
                 dynamic_shapes={}):
    """Build TensorRT Engine

    :use_fp16: set mixed flop computation if the platform has fp16.
    :use_dynamic_batch: use dynamic batch or not
    :max_batch_size: set max batch size. only use when use_dynamic_batch=True
    :dynamic_shapes: 
        BCHW中任意一个维度是动态的，需要设置本参数
        default {} represents not using dynamic.
        设置格式：{binding_name: (min_shape, opt_shape, max_shape)}, 例如：
        设置动态batch示例：{"input": ((1,3,224,224), (16,3,224,224), (16,3,224,224))}
        设置动态图片尺寸示例：{"input": ((1,3,224,224), (1,3,512,512), (1,3,768,768))}
    """
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)

    # create a network
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # importing model using the ONNX parser
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("===> Completed parsing ONNX file")

    # set optimize config
    config = builder.create_builder_config()
    #config.set_tactic_sources(trt.TacticSource.CUBLAS_LT)   # TODO ???
    # set workspace, default workspace is 2G
    config.max_workspace_size = 2 << 30
    # use fp16
    if builder.platform_has_fast_fp16 and use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        # if use fp16, set IO format be fp16 as well
        for input_idx in range(network.num_inputs):
            network.get_input(input_idx).dtype = trt.DataType.HALF

    # set max batch size
    if use_dynamic_batch:
        print(f"===> using dynamic batch, max batch size: {max_batch_size}")
        builder.max_batch_size = max_batch_size
    else:
        # fixed batch size
        print(f"===> using fix batch size 1")
        builder.max_batch_size = 1

    # set dynamic shape
    if len(dynamic_shapes) > 0:
        print(f"===> using dynamic shapes: {str(dynamic_shapes)}")
        profile = builder.create_optimization_profile()
        for binding_name, dynamic_shape in dynamic_shapes.items():
            min_shape, opt_shape, max_shape = dynamic_shape
            profile.set_shape(
                binding_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # Remove existing engine file
    if os.path.exists(engine_save_path):
        try:
            os.remove(engine_save_path)
        except Exception:
            print(f"Cannot remove existing file: {engine_save_path}")

    print("===> Creating Tensorrt Engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_save_path, "wb") as f:
        f.write(serialized_engine)
    print("===> Serialized Engine Saved at: ", engine_save_path)


if __name__ == "__main__":

    onnx_file_path = "test/res50_batch_dynamic.onnx"
    engine_save_path = "test/res50_onnx2trt_fp16.trt"
    use_fp16 = True
    use_dynamic_batch = True
    max_batch_size = 16
    dynamic_shapes = {"input": ((1,3,224,224), (16,3,224,224), (16,3,224,224))}

    build_engine(onnx_file_path, engine_save_path, 
                    use_fp16=use_fp16, 
                    use_dynamic_batch=use_dynamic_batch,
                    max_batch_size=max_batch_size, 
                    dynamic_shapes=dynamic_shapes)
