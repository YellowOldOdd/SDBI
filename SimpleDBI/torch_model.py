#!/usr/bin/env python
# coding: utf-8

import torch 
import logging 
import threading
from SimpleDBI.model import Model

logger = logging.getLogger('torch_model')
logger.setLevel(logging.ERROR)

class TorchModel(object) :

    def __init__(self, model_name, model_path) : 
        # 1. set device
        self.device = 'cpu' # 'cuda:0'
        if torch.cuda.device_count() > 0 :
            self.device = 'cuda:0'
        logger.warning('Torch device {}.'.format(self.device))

        self.name = model_name
        self.model = torch.jit.load(model_path).to(self.device).eval()

    def tensor_to_numpy(self, torch_tensor) : 
        if isinstance(torch_tensor, tuple) :
            output_list = []
            for t in torch_tensor :
                output_list.append(t.cpu().numpy())
            return tuple(output_list)
        else :
            return tuple([torch_tensor.cpu().numpy()])

    def numpy_to_tensor(self, np_arr) : 
        if isinstance(np_arr, tuple) :
            output_list = []
            for na in np_arr :
                output_list.append(torch.from_numpy(na).to(self.device))
            return tuple(output_list)
        elif isinstance(np_arr, torch.Tensor) :
            return tuple(torch.from_numpy(numpy_arr).to(self.device))
    
    def model_forward(self, args) :
        with torch.no_grad():
            output_tensor = self.model(*args)
        return output_tensor
    
    def forward(self, *args) :
        input_torch_tensor = self.numpy_to_tensor(args)
        output_torch_tensor = self.model_forward(input_torch_tensor)
        numpy_tensor = self.tensor_to_numpy(output_torch_tensor)
        return numpy_tensor
