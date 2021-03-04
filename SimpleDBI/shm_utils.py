#!/usr/bin/env python
# coding: utf-8

import collections
import functools
import os
import operator
import threading
import multiprocessing as mp
import numpy as np

ShmInfo = collections.namedtuple(
    'ShmInfo', ['name', 'shape', 'dtype']
)

def create_shm(tensor_infos, name_suffix = None) :
    shm_names = []
    shms = []
    shm_arrs = []
    for info in tensor_infos :
        shm_name = '{}_{}_{}'.format(
            info.name, os.getpid(), threading.currentThread().ident,
        )
        if name_suffix is not None :
            shm_name = '{}_{}'.format(shm_name, name_suffix)
        shm_size = \
            functools.reduce(operator.mul, info.shape) * \
            np.dtype(info.dtype).itemsize

        shm = mp.shared_memory.SharedMemory(
            name = shm_name, create = True, size=shm_size)
        shm_arr = np.ndarray(info.shape, dtype=info.dtype, buffer=shm.buf)

        shm_names.append(shm_name)
        shms.append(shm)
        shm_arrs.append(shm_arr)

    return shm_names, shms, shm_arrs


def get_shm(names, tensor_infos) :
    shms = []
    shm_arrs = []
    assert len(names) == len(tensor_infos)
    for name, info in zip(names, tensor_infos) :
        shm = mp.shared_memory.SharedMemory(name = name, create=False)
        shm_arr = np.ndarray(info.shape, dtype=info.dtype, buffer=shm.buf)

        shms.append(shm)
        shm_arrs.append(shm_arr)

    return shms, shm_arrs

def close_shm(shms, shm_arrs, unlink) :
    for arr in shm_arrs :
        del arr
    
    try :
        for shm in shms :
            shm.close()
            if unlink :
                shm.unlink()
    except :
        pass
