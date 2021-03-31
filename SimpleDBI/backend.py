#!/usr/bin/env python

import collections
import logging
import multiprocessing as mp
import numpy as np
import os
import queue
import threading
import traceback

from time import time
from SimpleDBI.backend_metric import BackendMetric
from SimpleDBI.shm_handler import ShmHandler, gen_name

EXIT_SIG = -1
MAX_SESSION_SLOT_NUM = 128
MAX_BATCHED_TENSOR_NUM = 32

logger = logging.getLogger('backend')

BatchedIndex = collections.namedtuple(
    'BatchedIndex', ['qid', 'length', 'metric']
)

def backend_process(entry_queue, args) : 
    backend = Backend(args)
    backend.start()
    backend.session_connector(entry_queue)
    backend.close()
    logger.debug('Backend process exit.')

def model_process(
        model_name, model_type, model_path, 
        shm_queue, conn, input_info, output_info) :
    # 1. init model
    if model_type == 'mock' :
        from SimpleDBI.mock_model import MockModel
        model = MockModel(model_name, model_path)
    elif model_type == 'torch' :
        from SimpleDBI.torch_model import TorchModel
        model = TorchModel(model_name, model_path)
    elif model_type == 'tf' :
        from SimpleDBI.tf_model import TFModel
        model = TFModel(model_name, model_path, input_info, output_info)
    else :
        logger.error('ERROR MODEL TYPE : {}'.format(model_type))

    logger.debug('model_process up')

    # 2. create shared memoty
    # 2.1 create output shared memory
    output_shm_name = []
    output_shm = []
    for info in output_info :
        shm_name = gen_name(info['name'])
        sh = ShmHandler(shm_name, info['max_shape'], info['dtype'])
        sh.create_shm()
        output_shm_name.append(shm_name)
        output_shm.append(sh)

    # 2.2 load input shared memory
    input_shm_name_list = conn.recv()
    input_shm_list = []
    for input_shm_name in input_shm_name_list :
        input_shm = []
        for shm_name, info in zip(input_shm_name, input_info) :
            sh = ShmHandler(shm_name, info['max_shape'], info['dtype'])
            sh.load_shm()
            input_shm.append(sh)
        input_shm_list.append(input_shm)

    conn.send(output_shm_name)

    # 3. inference
    while True :
        value = conn.recv()
        if value == EXIT_SIG :
            break
        
        # 3.1 load input 
        shm_idx, shapes = value
        inputs = []
        input_shm = input_shm_list[shm_idx]
        for shape, sh in zip(shapes, input_shm) :
            shm_arr = sh.ndarray(shape)
            # input_arr = shm_arr[:]
            # inputs.append(input_arr)
            inputs.append(shm_arr)

        # 3.2 forward
        outputs = model.forward(*inputs)

        # 3.3 write output
        shapes = []
        for output, sh in zip(outputs, output_shm) :
            shape = output.shape
            shm_arr = sh.ndarray(shape)
            shm_arr[:] = output[:]
            shapes.append(shape)

        conn.send(shapes)
        shm_queue.put(shm_idx) # send shared memory to avalible queue
    
    # 4. clean
    for input_shm in input_shm_list :
        for sh in input_shm :
            sh.close()

    conn.send(True)
    stat = conn.recv()
    assert stat
    for sh in output_shm :
        sh.close()

    conn.close()
    logger.debug('Model process exit.')

class Backend(object) : 

    def __init__(self, args) : 
        logger.debug('Create backend {}'.format(args['name']))

        self.name = args['name']
        
        self.input_tensor_queue   = queue.Queue(maxsize = MAX_SESSION_SLOT_NUM)
        self.batched_tensor_queue = queue.Queue(maxsize = MAX_BATCHED_TENSOR_NUM)
        self.output_tensor_queue  = queue.Queue(maxsize = MAX_SESSION_SLOT_NUM)

        self.dynamic_batch = args.get('dynamic_batch') 
        self.duplicate_num = args.get('duplicate_num') 
        self.model_path = args['path']
        self.model_type = args.get('model_type')
        self.max_batch_size = args.get('max_batch_size')
        self.timeout = args.get('timeout') 
        self.use_mps = args.get('use_mps') 

        self.threads = {}                     # all threads
        self.io_queues = []                   # io queue of request handler
        self.io_queue_lock = threading.Lock() # lock for create request handler

        # input shared memory
        self.input_shm_name_set = []              # shared memory name for concat and inference
        self.input_shm_set = []                   # shared memory for concat and inference
        self.input_shm_queue = mp.Queue(maxsize=3 * self.duplicate_num)
        
        self.input_info = args.get('input_info')
        # create a set of input shared memory
        for idx in range(3 * self.duplicate_num) :
            input_shm_name = [] 
            input_shm = []
            for info in self.input_info :
                shm_name = gen_name(info['name'], suffix = idx)
                sh = ShmHandler(shm_name, info['max_shape'], info['dtype'])
                sh.create_shm()
                input_shm_name.append(shm_name)
                input_shm.append(sh)

            self.input_shm_name_set.append(input_shm_name)
            self.input_shm_set.append(input_shm)
            self.input_shm_queue.put(idx)
        
        # output shared memory info
        self.output_info = args.get('output_info')

        self.use_mps = False      if self.use_mps is None else self.use_mps
        self.timeout = 0.001      if self.timeout is None else self.timeout
        self.dynamic_batch = True if self.dynamic_batch is None else self.dynamic_batch
        self.max_batch_size = 32  if self.max_batch_size is None else self.max_batch_size
        self.duplicate_num = 1    if self.duplicate_num is None else self.duplicate_num
        
    def __del__(self) :
        logger.debug('Backend {} quit'.format(self.name))

    def start(self) :
        self.alive = True
        for fname, func in { 
                'batcher' : self.batch_handler, 
                'scatter' : self.output_handler,
            }.items() :
            t = threading.Thread(target=func, )
            t.start()
            self.threads[fname] = t
        
        for idx in range(self.duplicate_num) : 
            t = threading.Thread(target=self.mps_model_handler, )
            t.setDaemon(True)
            t.start()
            self.threads['gpu_model_{}'.format(idx)] = t

    def close(self) :
        self.alive = False

        for name, t in self.threads.items() :
            logger.debug('joining {}'.format(name))
            t.join()
        logger.debug('All backend {} thread exit.'.format(self.name))

        for input_shm in self.input_shm_set :
            for sh in input_shm :
                sh.close()

    def request_handler(self, conn) :

        def get_tensor_info_from_session(create) :
            tensor_info = conn.recv()
            shm_list = []
            for info in tensor_info :
                sh = ShmHandler(info['shm_name'], info['max_shape'], info['dtype'])
                if create :
                    sh.create_shm()
                else :
                    sh.load_shm()
                shm_list.append(sh)

            conn.send(True)
            return shm_list

        # 1. get dtype and max size 
        input_shm = get_tensor_info_from_session(False)
        output_shm = get_tensor_info_from_session(True)

        # 3. get io id
        self.io_queue_lock.acquire()
        self.io_queues.append(queue.Queue(maxsize=1)) # like a thread pipe
        QID = len(self.io_queues) - 1
        self.io_queue_lock.release()

        # 4. listening for request tensor
        while True :
            value = conn.recv()

            # 4.1 handle exit signal
            time_metric = BackendMetric()
            time_metric.arrive = time()

            if value == EXIT_SIG :
                break
            shapes = value
            
            # 4.2 push input to queue
            inputs = []
            for shape, sh in zip(shapes, input_shm) :
                input_arr = sh.ndarray(shape)
                inputs.append(input_arr)
            
            time_metric.input_queue_put = time()
            self.input_tensor_queue.put((inputs, QID, time_metric))
            
            # 4.3 pop output from queue
            outputs, metric = self.io_queues[QID].get()
            shapes = []
            for output_arr, sh in zip(outputs, output_shm) :
                shape = output_arr.shape
                shm_arr = sh.ndarray(shape)
                shm_arr[:] = output_arr[:]
                shapes.append(shape)

            # 4.4 tell session the inference is finished
            metric.send = time()
            conn.send((shapes, metric))
            # print_metric(metric)
        
        # 5. clean
        try :
            for shm in output_shm :
                shm.close()
        except :
            pass

        # wait for remote shared memory cleaned
        conn.send(EXIT_SIG)
        conn.recv()

        # clean local shared memory
        try :
            for shm, hold_src in input_shm :
                shm.close()
        except :
            pass

        conn.close()
        # print('request_handler exit ...')
        
    def session_connector(self, process_queue) : 
        '''
        Build connection with local inference session
        '''
        conn_num = 0
        while True : 
            value = process_queue.get()
            if value == EXIT_SIG :
                # print('session_connector receive quit signal')
                break
            
            conn = value 

            # 1. connection built, ping back 
            conn.send(True)

            # 2. start dataloader thread
            req_thread = threading.Thread(
                target = self.request_handler, args = (conn, ))
            req_thread.setDaemon(True)
            req_thread.start()

            # self.threads.append(req_thread)
            self.threads['req_{}'.format(conn_num)] = req_thread
            conn_num +=1 

        logger.debug('{} session_connector quit.'.format(self.name))

    def batch_handler(self) :
        '''
        collect request and make batch
        '''
        latest_tensor = None
        latest_qid = None
        latest_metric = None

        while self.alive : 
            try :
                batch_size = 0    # batch size
                tensor_list = []  # batch data
                batch_index = []  # batch index

                def add_tensor(tensor : list , qid : int, metric : BackendMetric) :
                    # start = time()
                    nonlocal batch_size 
                    nonlocal tensor_list
                    nonlocal batch_index 
                    # 1. check all tensor has same batch size
                    bs = len(tensor[0])
                    assert bs <= self.max_batch_size
                    for t in tensor :
                        assert t.shape[0] == bs
                    
                    if bs + batch_size > self.max_batch_size :
                        return False
                    
                    # 2. init tensor_list
                    if len(tensor_list) == 0 :
                        for _ in range(len(tensor)) :
                            tensor_list.append([])

                    # 3. add data
                    batch_size += bs
                    for ts, ts_l in zip(tensor, tensor_list) :
                        ts_l.append(ts)
                    
                    batch_index.append(BatchedIndex(qid, bs, metric))
                    # print('add latency : {}'.format((time() - start)*1000))
                    return True
                
                # 1. append tensor
                if latest_tensor is not None :
                    assert add_tensor(latest_tensor, latest_qid, latest_metric) 
                while True :
                    try :
                        latest_tensor, latest_qid, latest_metric = \
                            self.input_tensor_queue.get(timeout = self.timeout)
                        latest_metric.input_queue_get = time()

                        # start = time()
                        if not add_tensor(latest_tensor, latest_qid, latest_metric) :
                            break
                        # print('add latency : {}'.format((time() - start)*1000))
                        
                    except queue.Empty :
                        latest_tensor, latest_qid = None, None
                        if batch_size > 0 :
                            break
                        else :
                            if self.alive :
                                continue            
                            else :
                                # print('batch_handler exit.')
                                return
                
                # 2. concat tensors
                start = time()
                shapes = []
                shm_idx = self.input_shm_queue.get()
                input_shm = self.input_shm_set[shm_idx]
                for tensors, sh in zip(tensor_list, input_shm) :
                    shape = list(tensors[0].shape)
                    shape[0] = batch_size
                    batch_data = sh.ndarray(shape)
                    np.concatenate(tensors, axis = 0, out = batch_data)
                    shapes.append(shape)

                concat_latency = time() - start

                # 3. push meta info to queue
                t = time()
                for index in batch_index :
                    index.metric.model_queue_put = t
                    index.metric.batch_size = batch_size
                    index.metric.concat = concat_latency
                self.batched_tensor_queue.put((shm_idx, shapes, batch_index))
            except :
                logger.error(traceback.format_exc())
            
    def output_handler(self) :
        while self.alive :
            try :
                batch_output, batch_index = \
                    self.output_tensor_queue.get(timeout=1)
            except queue.Empty :
                continue
                
            # metric
            t = time()
            for index in batch_index :
                index.metric.output_queue_get = t
            
            offset = 0
            for index in batch_index :
                outputs = []
                for tensor in batch_output :
                    shape = list(tensor.shape)
                    shape[0] = index.length
                    output = np.empty(shape, tensor.dtype)
                    output[:] = tensor[offset: offset + index.length]
                    outputs.append(output)
                offset += index.length
                index.metric.backend_end = time()
                self.io_queues[index.qid].put((outputs, index.metric))

        logger.debug('output_handler exit.')

    def mps_model_handler(self) :
        # 1. create backend model process
        conn_backend, conn_model = mp.Pipe()
        proc = mp.Process(
            target = model_process,
            args = (self.name, self.model_type, self.model_path, 
                    self.input_shm_queue , conn_model, 
                    self.input_info, self.output_info))
        proc.start()

        # 2. create shared memory
        conn_backend.send(self.input_shm_name_set)

        output_shm_name = conn_backend.recv()
        output_shm = []
        for shn_name, info in zip(output_shm_name, self.output_info) :
            sh = ShmHandler(shn_name, info['max_shape'], info['dtype'])
            sh.load_shm()
            output_shm.append(sh)

        # 3. inference
        while self.alive :
            try :
                shm_idx, shapes, batch_index = \
                    self.batched_tensor_queue.get(timeout=1)
            except queue.Empty :
                continue

            t = time()
            for index in batch_index :
                index.metric.model_queue_get = t

            conn_backend.send((shm_idx, shapes))
            shapes = conn_backend.recv()

            batch_output = []
            for shape, sh in zip(shapes, output_shm) :
                shm_arr = sh.ndarray(shape)
                output_arr = np.empty(shape, shm_arr.dtype)
                output_arr[:] = shm_arr[:]
                batch_output.append(output_arr)

            for index in batch_index :
                index.metric.output_queue_put = t
            self.output_tensor_queue.put((batch_output, batch_index))
        
        # 4. clean 
        conn_backend.send(EXIT_SIG)
        stat = conn_backend.recv()
        for sh in output_shm :
            sh.close()
        conn_backend.send(True)

        proc.join()
