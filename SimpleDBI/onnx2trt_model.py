#!/usr/bin/env python
# coding: utf-8

import logging 
import traceback
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
#import pycuda.autoinit

logger = logging.getLogger('tensorRT_model')
logger.setLevel(logging.ERROR)

TRT_LOGGER = trt.Logger()


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
                self.host = host_mem
                self.device = device_mem

        def __str__(self):
                return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
                return self.__str__()


class TensorRTModel(object):

    def __init__(self, model_name, model_path):

        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()

        self.model_name = model_name
        self.engine = None
        self.engine_file = model_path

        self.bindings = None
        self.stream = None
        self.context = None
        self.max_batch_size = None

        self.inputs = None
        self.input_shapes = None
        self.input_names = None
        self.outputs = None
        self.out_shapes = None
        self.out_names = None

        # deserialize tensorrt engine
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        # allocate memory buffers before inference
        self.bindings, self.stream, self.max_batch_size, \
        self.inputs, self.input_shapes, self.input_names, \
        self.outputs, self.out_shapes, self.out_names = self._allocate_buffers(self.engine)

        # create context
        self.context = self.engine.create_execution_context()


    def forward(self, *args) :
        """
        推理接口

        *args: 数量不固定的模型输入
               each parameter is a ndarray, shape format is NCHW, like [16, 3, 224, 224]
               多数情况下模型只有一个输入参数，即带batch的图片输入
               这样设计是为了兼容多输入的复杂模型
        """
        try:
            self.cuda_ctx.push()
            batch_size = args[0].shape[0]

            for idx, cur_input in enumerate(args):
                # set context binding shape
                # 由于我们是dynamic batch，每次输入的batch可能不同，因此每次推理都要设置
                cur_binding_idx = self.engine.get_binding_index(self.input_names[idx])
                self.context.set_binding_shape(cur_binding_idx, cur_input.shape)
                # assign value for host memory
                self.inputs[idx].host = np.ascontiguousarray(cur_input)

            trt_outputs = self._inference(
                self.context, bindings=self.bindings,
                inputs=self.inputs, outputs=self.outputs, stream=self.stream)

            # collect trt outputs
            trt_outputs = []
            #for cur_output, cur_ouput_shape in zip(self.outputs, self.out_shapes):
            for idx in range(len(self.outputs)):
                cur_output_host = self.outputs[idx].host
                cur_output_shape = self.out_shapes[idx]
                # reshape TRT outputs to original shape instead of flattened array
                cur_output = cur_output_host.reshape(cur_output_shape)
                # 获取有效部分 ouput[batch_size+1:max_batch_size]的部分是无效的
                cur_output = cur_output[:batch_size]
                trt_outputs.append(cur_output)
            trt_outputs = tuple(trt_outputs)

        except:
            print(traceback.format_exc())
        finally:
            self.cuda_ctx.pop()

        return trt_outputs


    def _inference(self, context, bindings, inputs, outputs, stream):
        """
        inference function

        inputs and outputs are expected to be lists of HostDeviceMem objects.
        This function is generalized for multiple inputs/outputs.
        """
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]


    def _allocate_buffers(self, engine):
        # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        out_shapes = []
        input_shapes = []
        out_names = []
        input_names = []

        max_batch_size = engine.max_batch_size
        for binding in engine:
            # get binding_shape (value == -1 means dynamic shape)
            binding_shape = engine.get_binding_shape(binding)
            # compute max_size and dtype
            size = abs(trt.volume(binding_shape)) * max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # collect info to appropriate list
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
                input_shapes.append(binding_shape)
                input_names.append(binding)
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                out_shapes.append(binding_shape)
                out_names.append(binding)
        return bindings, stream, max_batch_size, inputs, input_shapes, input_names, outputs, out_shapes, out_names


    def __del__(self):
        self.cuda_ctx.pop()
        del self.cuda_ctx

