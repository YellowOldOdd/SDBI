#!/usr/bin/env python

import numpy as np
import os
import torch
from SimpleDBI.session import Session, Run

os.environ['KMP_DUPLICATE_LIB_OK']='True'
model_name = 'test.jit'

def forward(max_batch_size, np_arr) :
    # np_arr = np.random.random(
    #     size = (1,3,224,224)).astype(np.float32)

    # 1. run SDBI session
    sess = Session(
        name            = 'test', 
        path            = model_name, 
        max_batch_size  = max_batch_size,
        input_info      = [{
            'name' : 'input1',
            'max_shape' : [max_batch_size, 3, 224, 224],
            'dtype' : np.float32,
        },], 
        output_info     = [{
            'name' : 'output1',
            'max_shape' : [max_batch_size, 2],
            'dtype' : np.float32,
        },], 
    )
    sdbi_output = sess.forward(np_arr)
    print('batching output : {}'.format(sdbi_output))

def main() :
    max_batch_size = 2
    
    np_arr = np.random.random(
        size = (1,3,224,224)).astype(np.float32)
    tensor = torch.from_numpy(np_arr)
    jit_model = torch.jit.load(model_name).to('cpu').eval()
    torch_output = jit_model.forward(tensor)
    print('pytorch output  : {}'.format(torch_output))

    Run(target = forward, worker_num=10, args = (max_batch_size, np_arr))
    

if __name__ == '__main__' :
    main()