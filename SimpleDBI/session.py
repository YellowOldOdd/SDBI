#!/usr/bin/env python
# coding: utf-8

import functools 
import logging
import os
import numpy as np
import threading
import operator
from time import sleep
import multiprocessing as mp

from SimpleDBI.context import EXIT_SIG, context_server
from SimpleDBI.shm_handler import ShmHandler, gen_name

logger = logging.getLogger('session')
inference_context_queue = None

def set_context_queue(ctx_q) :
    global inference_context_queue 
    inference_context_queue = ctx_q

class Session(object) :
    def __init__(self, 
                name : str, 
                path : str,
                input_info : list,
                output_info : list,
                dynamic_batch : bool = True, 
                duplicate_num : int = 1,
                model_type : str = 'torch',
                max_batch_size : int = 32, 
                timeout : float = 0.003,
                metric : bool = False,
    ) : 
        global inference_context_queue 
        assert inference_context_queue 
        logger.debug('Session started.')

        # 1. backend params
        self.name = name
        backend_args = {
            'name'            : name,
            'path'            : path,
            'dynamic_batch'   : dynamic_batch,
            'duplicate_num'   : duplicate_num,
            'model_type'      : model_type,
            'max_batch_size'  : max_batch_size,
            'timeout'         : timeout,
            'input_info'      : input_info,
            'output_info'     : output_info,
        }
        
        # 2. build connect with backend
        self.conn_s, self.conn_c = mp.Pipe()
        inference_context_queue.put((self.conn_c, backend_args))

        stat = self.conn_s.recv()
        assert stat is True

        # 3. share memory with backend
        self.input_shm = []
        self.output_shm = []
        
        def _shm_info(tensor_infos) :
            for info in tensor_infos :
                assert info.get('name')
                assert info.get('max_shape')
                assert info.get('dtype')
                info['shm_name'] = gen_name(info['name'])
                info['shm_size'] = \
                    functools.reduce(operator.mul, info.get('max_shape')) * \
                    np.dtype(info.get('dtype')).itemsize
        
        _shm_info(input_info)
        for info in input_info :
            sh = ShmHandler(info['shm_name'], info['max_shape'], info['dtype'])
            sh.create_shm()
            self.input_shm.append(sh)
        self.conn_s.send(input_info)
        assert self.conn_s.recv()

        # load output shm
        _shm_info(output_info)
        self.conn_s.send(output_info)
        assert self.conn_s.recv()
        for info in output_info :
            sh = ShmHandler(info['shm_name'], info['max_shape'], info['dtype'])
            sh.load_shm()
            self.output_shm.append(sh)

    def __del__(self) :
        self.conn_s.send(EXIT_SIG)
        try :
            for shm in self.input_shm :
                shm.close()
        except :
            pass

        self.conn_s.recv()
        self.conn_s.send(EXIT_SIG)
        try :
            for shm in self.output_shm :
                shm.close()
        except :
            pass

    def forward(self, *args) : 
        shapes = []
        for tensor, sh in zip(args, self.input_shm) : 
            shape = list(tensor.shape)
            shm_arr = sh.ndarray(shape)
            shm_arr[:] = tensor[:]
            shapes.append(shape)
        
        self.conn_s.send(shapes)
        shapes = self.conn_s.recv()

        outputs = []
        for shape, sh in zip(shapes, self.output_shm) : 
            shape = list(shape)
            shm_arr = sh.ndarray(shape)
            tensor = np.empty(shape, shm_arr.dtype)
            tensor[:] = shm_arr[:]
            outputs.append(tensor)

        result = outputs[0] if len(outputs) == 1 else tuple(outputs)
        
        return result

def target_wrapper(ctx_q, target, args, kwargs) :
    set_context_queue(ctx_q)
    target(*args, **kwargs)

def Run(target, worker_num, metric_queue = None, args = (), kwargs = {}) :
    ctx_q = mp.Queue()
    ctx_process = mp.Process(target=context_server, args=(ctx_q, metric_queue))
    ctx_process.start()

    sess_process = []
    for _ in range(worker_num) :
        process = mp.Process(target=target_wrapper, args=(ctx_q, target, args, kwargs))
        process.start()
        sess_process.append(process)
        
    for p in sess_process :
        p.join()

    ctx_q.put(EXIT_SIG)
    ctx_process.join()
