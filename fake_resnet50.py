#!/usr/bin/env python

import numpy as np
import torch
import torchvision 
from torch2trt import torch2trt

class SimpleModel(torch.nn.Module) : 
    def __init__(self) : 
        super(SimpleModel, self).__init__()
        # 这里要下载ImageNet预训练模型，及pretrained=True
        # 这样可以使用图片测试出一些虽不报错但推理结果不对这种比较隐蔽的bug
        # 例如测试图片banana.jpg输出的类别索引应该为954
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.softmax = torch.nn.Softmax(dim = 1)
    
    def forward(self, x ) : 
        out = self.resnet50(x)
        out = self.softmax(out)
        return out

def main() :

    model = SimpleModel().eval()
    
    batch_size = 16

    # save jit model
    dummy_input = np.random.random(
        size = (batch_size,3,224,224)).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_input)
    jit_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(jit_model, "test/res50.jit")

    # save torch2trt model
    # convert to TensorRT feeding sample data as input
    dummy_input = torch.ones((batch_size, 3, 224, 224)).cuda()
    model = model.cuda()
    model_torch2trt_fp32 = torch2trt(model, [dummy_input], max_batch_size=batch_size, fp16_model=False, max_workspace_size=1<<27)
    torch.save(model_torch2trt_fp32.state_dict(), 'test/res50_torch2trt_fp32.pth')
    model_torch2trt_fp16 = torch2trt(model, [dummy_input], max_batch_size=batch_size, fp16_model=True, max_workspace_size=1<<27)
    torch.save(model_torch2trt_fp16.state_dict(), 'test/res50_torch2trt_fp16.pth')

    # save onnx model
    # this is a dynamic batch onnx convert demo
    onnx_save_path = 'test/res50_batch_dynamic.onnx'
    dynamic_axes = {'input': {0: "batch"},  # variable lenght axes
                    'output': {0: "batch"}}
    # 模型输入、输出的名称
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model,  # model being run
                      dummy_input,               # model input (or a tuple for multiple inputs)
                      onnx_save_path,            # where to save the onnx model
                      dynamic_axes=dynamic_axes,
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=input_names,   # the model's input names
                      output_names=output_names  # the model's output names
                      )
    # CHECK 转换的是否为动态尺寸
    import onnx
    model = onnx.load(onnx_save_path)
    print(model.graph.input[0].type.tensor_type.shape)

if __name__ == '__main__' :
    main()
