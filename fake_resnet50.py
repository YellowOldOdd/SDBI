#!/usr/bin/env python

import numpy as np
import torch
import torchvision 

model_name = 'test.jit'

class SimpleModel(torch.nn.Module) : 
    def __init__(self) : 
        super(SimpleModel, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=False, num_classes=2)
        self.softmax = torch.nn.Softmax(dim = 1)
    
    def forward(self, x ) : 
        out = self.resnet50(x)
        out = self.softmax(out)
        return out

def main() :
    model = SimpleModel()
    
    dummy_input = np.random.random(
        size = (1,3,224,224)).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_input)

    jit_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(jit_model, model_name)

if __name__ == '__main__' :
    main()
