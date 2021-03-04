#!/usr/bin/env python
# coding: utf-8

import os
import threading
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

# https://docs.python.org/3/library/multiprocessing.shared_memory.html

def producer(conn) :
    # os.environ["PYTHONWARNINGS"] = "ignore"
    feed_shm_name = '{}_{}_{}'.format(
        'test', os.getpid(), threading.currentThread().ident)
    print('input shm name : {}'.format(feed_shm_name))

    feed_shm = SharedMemory(
        name = feed_shm_name, create=True, size=2 * 4)
    
    feed_shm_arr = np.ndarray((1, 2), dtype=np.float32, buffer=feed_shm.buf)
    input_arr = np.random.random((1, 2)).astype(np.float32)
    feed_shm_arr[:] = input_arr[:]

    conn.send(feed_shm_name)
    result_shm_name = conn.recv()
    result_shm = SharedMemory(name = result_shm_name)
    result_shm_arr = np.ndarray((1, 2), dtype=np.float32, buffer=result_shm.buf)
    print('Output array : {}'.format(result_shm_arr))

    conn.send('exit')
    del result_shm_arr
    result_shm.close()
    
    conn.recv()
    del feed_shm_arr
    feed_shm.close()
    feed_shm.unlink()
    
    print('clean and exit')

    return

def consumer(conn) :
    # os.environ["PYTHONWARNINGS"] = "ignore"
    result_shm_name = '{}_{}_{}'.format(
        'test', os.getpid(), threading.currentThread().ident)
    print('output shm name : {}'.format(result_shm_name))
    result_shm = mp.shared_memory.SharedMemory(
        name = result_shm_name, create=True, size=2 * 4)
    result_shm_arr = np.ndarray((1, 2), dtype=np.float32, buffer=result_shm.buf)

    feed_shm_name = conn.recv()
    feed_shm = mp.shared_memory.SharedMemory(name = feed_shm_name)
    feed_shm_arr = np.ndarray((1, 2), dtype=np.float32, buffer=feed_shm.buf)

    print('Input array : {}'.format(feed_shm_arr))

    result_shm_arr[:] = feed_shm_arr[:] # fake inference ...
    conn.send(result_shm_name)

    conn.recv()
    del feed_shm_arr
    feed_shm.close()
    
    conn.send('exit')
    del result_shm_arr
    result_shm.close()
    result_shm.unlink()
    
    print('clean and exit')

    return

def main() :
    conn_c, conn_p = mp.Pipe()
    produce_proc = mp.Process(target=producer, args=(conn_p,), daemon = True)
    consume_proc = mp.Process(target=consumer, args=(conn_c,), daemon = True)

    produce_proc.start()
    consume_proc.start()

    produce_proc.join()
    consume_proc.join()

if __name__ == '__main__' :
    main()
