#!/usr/bin/env python
# coding: utf-8

import traceback
import functools
import os
import operator
import threading
import multiprocessing as mp
import numpy as np

from multiprocessing import shared_memory

def gen_name(name, suffix = None) : 
    shm_name = '{}_{}_{}'.format(
        name, os.getpid(), threading.currentThread().ident,
    )
    if suffix is not None :
        shm_name = '{}_{}'.format(shm_name, suffix)
    return shm_name

class ShmHandler(object) : 
    def __init__(self, name, shape, dtype) : # , suffix = '') :
        self.name = name
        self.dtype = dtype
        self.capacity = \
            functools.reduce(operator.mul, shape) * np.dtype(dtype).itemsize
        self.shm = None
        self.holder = False

    def create_shm(self) : 
        self.shm = shared_memory.SharedMemory(
            name=self.name, create=True, size=self.capacity)
        self.holder = True
    
    def load_shm(self) : 
        self.shm = shared_memory.SharedMemory(name=self.name, create=False)
        self.holder = False

    def ndarray(self, shape) :
        return np.ndarray(shape, dtype=self.dtype, buffer=self.shm.buf)

    def close(self) :
        try :
            self.shm.close()
            if self.holder :
                self.shm.unlink()
        except :
            logger.error('Shared Memory unlink error: {}'.format(
                traceback.format_exc()))
