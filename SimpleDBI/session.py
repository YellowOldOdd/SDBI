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
        self.shm = []
        self.input_shm_arr = []
        self.output_shm_arr = []
        self.input_info = input_info
        self.output_info = output_info
        self.metric = metric
        
        def _shm_info(tensor_infos) :
            for info in tensor_infos :
                assert info.get('name')
                assert info.get('max_shape')
                assert info.get('dtype')

                shm_name = '{}_{}_{}'.format(
                    info['name'], os.getpid(), threading.currentThread().ident,
                )
                shm_size = \
                    functools.reduce(operator.mul, info['max_shape']) * \
                    np.dtype(info['dtype']).itemsize
                
                info['shm_name'] = shm_name
                info['shm_size'] = shm_size
        
        # create input shm
        _shm_info(self.input_info)
        for info in self.input_info :
            shm = mp.shared_memory.SharedMemory(
                name = info['shm_name'], create=True, size=info['shm_size'])
            self.shm.append((shm, True))
            shm_arr = np.ndarray(info['max_shape'], dtype=info['dtype'], buffer=shm.buf)
            self.input_shm_arr.append(shm_arr)
        self.conn_s.send(self.input_info)
        assert self.conn_s.recv()

        # load output shm
        _shm_info(self.output_info)
        self.conn_s.send(self.output_info)
        assert self.conn_s.recv()
        for info in self.output_info :
            shm = mp.shared_memory.SharedMemory(
                name = info['shm_name'], create=False, size=info['shm_size'])
            self.shm.append((shm, False))
            shm_arr = np.ndarray(info['max_shape'], dtype=info['dtype'], buffer=shm.buf)
            self.output_shm_arr.append(shm_arr)

        # logger.error('Session created ...')

    def __del__(self) :
        os.environ["PYTHONWARNINGS"] = "ignore"
        self.conn_s.send(EXIT_SIG)

        try :
            for shm, hold_resource in self.shm :
                if not hold_resource :
                    shm.close()
        except :
            pass

        self.conn_s.recv()
        self.conn_s.send(EXIT_SIG)

        try :
            for shm, hold_resource in self.shm :
                if hold_resource :
                    shm.close()
                    shm.unlink()
        except :
            pass
        os.unsetenv('PYTHONWARNINGS')
        # print('Session {} exit.'.format(self.name))

    def forward(self, *args) : 
        shapes = []
        for tensor, shm_arr in zip(args, self.input_shm_arr) : 
            shape = list(tensor.shape)

            shm_arr = shm_arr.reshape([-1] + shape[1:])
            shm_arr[:shape[0]] = tensor[:]
            shapes.append(shape)
        
        self.conn_s.send(shapes)
        shapes, metric = self.conn_s.recv()
        # print_metric(metric)

        outputs = []
        for shape, shm_arr in zip(shapes, self.output_shm_arr) : 
            shape = list(shape)
            shm_arr = shm_arr.reshape([-1] + shape[1:])
            tensor = np.empty(shape, shm_arr.dtype)
            tensor[:] = shm_arr[:shape[0]]
            outputs.append(tensor)

        result = outputs[0] if len(outputs) == 1 else tuple(outputs)

        if self.metric :
            return result, metric
        
        return result

def target_wrapper(ctx_q, target, args, kwargs) :
    set_context_queue(ctx_q)
    target(*args, **kwargs)

def Run(target, worker_num, args=(), kwargs = {}) :
    ctx_q = mp.Queue()
    ctx_process = mp.Process(target=context_server, args=(ctx_q, ))
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
