import numpy as np
import os
import torch
from SimpleDBI.session import Session, Run

model_name = './frozen_resnet_v1_50.pb'

def forward(max_batch_size, np_arr) :
    # 1. run SDBI session
    sess = Session(
        name            = 'resnet_v1_50', 
        path            = model_name, 
        max_batch_size  = max_batch_size,
        input_info      = [{
            'name' : 'input',
            'max_shape' : [max_batch_size, 3, 224, 224],
            'dtype' : np.float32,
        },], 
        output_info     = [{
            'name' : 'resnet_v1_50-predictions-Reshape_1',
            'max_shape' : [max_batch_size, 1000],
            'dtype' : np.float32,
        },], 
        model_type      = "tf",
    )
    sdbi_output = sess.forward(np_arr)

def main() :
    max_batch_size = 2 
    np_arr = np.random.random(
        size = (1,3,224,224)).astype(np.float32)
    Run(target = forward, worker_num=3, args = (max_batch_size, np_arr))
    

if __name__ == '__main__' :
    main()