#!/usr/bin/env python

import collections
import logging
import multiprocessing as mp
import numpy as np
import os
import queue
import threading
import traceback

from time import time, sleep
from SimpleDBI.shm_handler import ShmHandler, gen_name

EXIT_SIG = -1
MAX_SESSION_SLOT_NUM = 128
MAX_BATCHED_TENSOR_NUM = 32

logger = logging.getLogger('backend')

BatchedIndex = collections.namedtuple(
    'BatchedIndex', ['qid', 'length']
)

def backend_process(entry_q, metric_q, args) : 
    try :
        if metric_q is not None :
            args['metric_queue'] = metric_q
        backend = Backend(args)
        backend.start()
        backend.session_connector(entry_q)
        backend.close()
        logger.debug('Backend process exit.')
    except :
        logger.error('backend_process initialize error')
        logger.error(traceback.format_exc())

def model_process(
        model_name, model_type, model_path, 
        shm_queue, conn, input_info, output_info, pid, metric_q) :
    try :
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
            raise RuntimeError('ERROR MODEL TYPE : {}'.format(model_type))

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
    except :
        logger.error('model_process initialize error')
        logger.error(traceback.format_exc())
        return 
    logger.error('model_process <{}> initialize done'.format(model_name))

    tags = {'model' : '{}_{}'.format(model_name, pid)}
    # 3. inference
    while True :
        value = conn.recv()
        if value == EXIT_SIG :
            break
        
        shm_idx, input_shapes = value
        inputs = []
        output_shapes = []
        try :
            ts = time()
            # 3.1 load input 
            input_shm = input_shm_list[shm_idx]
            for shape, sh in zip(input_shapes, input_shm) :
                shm_arr = sh.ndarray(shape)
                inputs.append(shm_arr)

            # 3.2 forward
            outputs = model.forward(*inputs)

            # 3.3 write output
            for output, sh in zip(outputs, output_shm) :
                shape = output.shape
                shm_arr = sh.ndarray(shape)
                shm_arr[:] = output[:]
                output_shapes.append(shape)

            if metric_q is not None :
                metric_q.put({
                    "tags"   : tags,
                    "fields" : {'model_proc_cost' : time() - ts},
                })
                

        except :
            logger.error('model_process runtime error')
            logger.error(traceback.format_exc())

        finally :
            conn.send(output_shapes)
            shm_queue.put(shm_idx) # send shared memory to avalible queue

    
    # 4. clean
    try :
        for input_shm in input_shm_list :
            for sh in input_shm :
                sh.close()

        conn.send(True)
        stat = conn.recv()
        assert stat
        for sh in output_shm :
            sh.close()

        conn.close()
    except :
        logger.error('model_process destructor error')
        logger.error(traceback.format_exc())

    logger.error('Model process exit.')
    

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

        self.metric_q = args.get('metric_queue')
        self.tags = {'model' : self.name}

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
        self.dynamic_batch = True if self.dynamic_batch is None else self.dynamic_batch
        self.max_batch_size = 32  if self.max_batch_size is None else self.max_batch_size
        self.duplicate_num = 1    if self.duplicate_num is None else self.duplicate_num

        self.adapt = False
        if self.timeout is None :
            # print('TIMEOUT IS NONE')
            self.timeout = 0.01 
            self.adapt = True
        
    def __del__(self) :
        logger.debug('Backend {} quit'.format(self.name))
    
    def emit_metric(self, points, tag = None) :
        mtags = self.tags
        if tag is not None :
            mtags.update(tag)
        if self.metric_q is not None :
            self.metric_q.put({
                "tags"        : mtags,
                "fields"      : points,
            })

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
            t = threading.Thread(target=self.mps_model_handler, args=(idx, ))
            t.setDaemon(True)
            t.start()
            self.threads['gpu_model_{}'.format(idx)] = t

        t = threading.Thread(target=self.qmonitor, )
        t.setDaemon(True)
        t.start()
        self.threads['qmonitor'] = t


    def close(self) :
        self.alive = False

        for name, t in self.threads.items() :
            logger.debug('joining {}'.format(name))
            t.join()
        logger.debug('All backend {} thread exit.'.format(self.name))

        for input_shm in self.input_shm_set :
            for sh in input_shm :
                sh.close()
    
    def qmonitor(self) : 
        while self.alive : 
            end = float(int(time()) + 1)
            while time() < end :
                sleep(0.1)
                self.emit_metric(
                    {'input_tensor_qsize_value' : self.input_tensor_queue.qsize()})
                self.emit_metric(
                    {'batched_tensor_qsize_value' : self.batched_tensor_queue.qsize()})
                self.emit_metric(
                    {'output_tensor_qsize_value' : self.output_tensor_queue.qsize()})

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

        try :
            # 1. get dtype and max size 
            input_shm = get_tensor_info_from_session(False)
            output_shm = get_tensor_info_from_session(True)

            # 2. get io id
            self.io_queue_lock.acquire()
            self.io_queues.append(queue.Queue(maxsize=1)) # like a thread pipe
            QID = len(self.io_queues) - 1
            self.io_queue_lock.release()
        except :
            logger.error('request_handler initialize error')
            logger.error(traceback.format_exc())
            return 

        # 3. listening for request tensor
        while True :
            value = conn.recv()
            start_ts = time()

            # 3.1 handle exit signal
            if value == EXIT_SIG :
                break
            input_shapes = value
            output_shapes = []

            if len(input_shapes) < 1 :
                return None

            try :
                # 3.2 push input to queue
                inputs = []
                for shape, sh in zip(input_shapes, input_shm) :
                    if shape[0] > self.max_batch_size :
                        self.emit_metric({'batchsize_error_counter' : 1})
                        raise RuntimeError(
                            'Batch size {} > max_batch_size({}).'.format(
                                shape[0], self.max_batch_size))
                    input_arr = sh.ndarray(shape)
                    inputs.append(input_arr)
                
                self.input_tensor_queue.put((inputs, QID, time()))
                
                # 3.3 pop output from queue
                outputs = self.io_queues[QID].get()
                for output_arr, sh in zip(outputs, output_shm) :
                    shape = output_arr.shape
                    shm_arr = sh.ndarray(shape)
                    shm_arr[:] = output_arr[:]
                    output_shapes.append(shape)

                # 3.4 tell session the inference is finished
                self.emit_metric({
                    'forward_cost' : time() - start_ts,
                    'request_counter' : input_shapes[0][0],
                })
            except :
                self.emit_metric({'request_handler_error_counter' : 1})
                logger.error('request_handler runtime error')
                logger.error(traceback.format_exc())
            finally :
                conn.send((output_shapes))
        
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
            
            try :
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
            
            except :
                logger.error('session_connector runtime error.')
                logger.error(traceback.format_exc())

        logger.debug('{} session_connector quit.'.format(self.name))

    def batch_handler(self) :
        '''
        collect request and make batch
        '''
        latest_tensor = None
        latest_qid = None

        while self.alive : 
            try :
                batch_size = 0    # batch size
                tensor_list = []  # batch data
                batch_index = []  # batch index

                def add_tensor(tensor : list , qid : int, ) :
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
                    
                    batch_index.append(BatchedIndex(qid, bs))
                    # print('add latency : {}'.format((time() - start)*1000))
                    return True
                
                # 1. append tensor
                start_ts = time()

                if latest_tensor is not None :
                    assert add_tensor(latest_tensor, latest_qid,) 
                
                end = time() + self.timeout
                while True :
                    if time() > end and batch_size > 1 : 
                        latest_tensor, latest_qid = None, None
                        break
                    try :
                        latest_tensor, latest_qid, arrive_ts = \
                            self.input_tensor_queue.get(timeout = self.timeout)
                        # self.emit_metric({'backend_input_queue_cost' : time() - arrive_ts})

                        if not add_tensor(latest_tensor, latest_qid,) :
                            break
                        
                    except queue.Empty :
                        latest_tensor, latest_qid = None, None
                        if batch_size > 0 :
                            break
                        else :
                            if self.alive :
                                continue            
                            else :
                                logger.debug('batch_handler exit.')
                                return
                
                self.emit_metric({'backend_batch_gather_cost' : time() - start_ts})
                
                # 2. concat tensors
                start_ts = time()
                shapes = []
                shm_idx = self.input_shm_queue.get()
                input_shm = self.input_shm_set[shm_idx]
                for tensors, sh in zip(tensor_list, input_shm) :
                    shape = list(tensors[0].shape)
                    shape[0] = batch_size
                    batch_data = sh.ndarray(shape)
                    np.concatenate(tensors, axis = 0, out = batch_data)
                    shapes.append(shape)
                
                self.emit_metric({'backend_batch_size_value' : batch_size})
                self.emit_metric({'backend_batch_concat_cost' : time() - start_ts})
                # 3. push meta info to queue
                self.batched_tensor_queue.put((shm_idx, shapes, batch_index, time()))
            except :
                self.emit_metric({'batch_handler_error_counter' : 1})
                logger.error('batch_handler runtime error')
                logger.error('may cause unkonwn behavior...')
                logger.error(traceback.format_exc())
            
    def output_handler(self) :
        while self.alive :
            try :
                batch_output, batch_index = \
                    self.output_tensor_queue.get(timeout=1)
            except queue.Empty :
                continue
            
            start_ts = time()
            offset = 0
            for index in batch_index :
                outputs = []
                try :
                    for tensor in batch_output :
                        shape = list(tensor.shape)
                        shape[0] = index.length
                        output = np.empty(shape, tensor.dtype)
                        output[:] = tensor[offset: offset + index.length]
                        outputs.append(output)
                    offset += index.length
                    
                except : 
                    self.emit_metric({'output_handler_error_counter' : 1})
                    logger.error('output_handler runtime error.')        
                    logger.error(traceback.format_exc())

                finally :
                    self.io_queues[index.qid].put((outputs))
            
            self.emit_metric({'backend_scatter_cost' : time() - start_ts})

        logger.debug('output_handler exit.')

    def mps_model_handler(self, idx) :
        try :
            # 1. create backend model process
            conn_backend, conn_model = mp.Pipe()
            proc = mp.Process(
                target = model_process,
                args = (self.name, self.model_type, self.model_path, 
                        self.input_shm_queue , conn_model, 
                        self.input_info, self.output_info, idx, self.metric_q))
            proc.start()

            # 2. create shared memory
            conn_backend.send(self.input_shm_name_set)

            output_shm_name = conn_backend.recv()
            output_shm = []
            for shn_name, info in zip(output_shm_name, self.output_info) :
                sh = ShmHandler(shn_name, info['max_shape'], info['dtype'])
                sh.load_shm()
                output_shm.append(sh)
        except :
            logger.error('mps_model_handler initialize error')
            logger.error(traceback.format_exc())
            return
        
        def health_check() :
            while True :
                sleep(5)
                tag = {'model_handler_name' : '{}_{}'.format(self.name, idx)}
                if proc.is_alive() :
                    self.emit_metric({'model_handler_health_value' : 1}, tag = tag)
                else :
                    self.emit_metric({'model_handler_health_value' : 0}, tag = tag)
        
        health_thread = threading.Thread(target=health_check, daemon=True)
        health_thread.start()

        # 3. inference
        while self.alive :
            start_ts = time()
            try :
                shm_idx, shapes, batch_index, batch_q_ts = \
                    self.batched_tensor_queue.get(timeout=1)
            except queue.Empty :
                continue
            except :
                logger.error('mps_model_handler error')
                logger.error(traceback.format_exc())
            
            batch_output = []
            try :
                model_start_ts = time()
                conn_backend.send((shm_idx, shapes))
                shapes = conn_backend.recv()
                self.emit_metric({'backend_forward_model_cost' : time() - model_start_ts})

                for shape, sh in zip(shapes, output_shm) :
                    shm_arr = sh.ndarray(shape)
                    output_arr = np.empty(shape, shm_arr.dtype)
                    output_arr[:] = shm_arr[:]
                    batch_output.append(output_arr)

                fwd_cost = time() - start_ts
                self.emit_metric({'backend_forward_cost' : fwd_cost})

                if self.adapt :
                    delta = fwd_cost / (0.5 + self.duplicate_num) - self.timeout
                    if abs(delta) / self.timeout > 0.2 :
                        self.io_queue_lock.acquire()
                        self.timeout = self.timeout * 0.8 + (self.timeout + delta) * 0.2
                        self.io_queue_lock.release()
                        # print('forward cost : {}, timeout : {}'.format(
                        #     fwd_cost, self.timeout
                        # ))

            except :
                logger.error('mps_model_handler error')
                logger.error(traceback.format_exc())
                self.emit_metric({'mps_model_handler_error_counter' : 1})

            finally :
                self.output_tensor_queue.put((batch_output, batch_index))

        
        # 4. clean 
        conn_backend.send(EXIT_SIG)
        stat = conn_backend.recv()
        for sh in output_shm :
            sh.close()
        conn_backend.send(True)

        proc.join()
