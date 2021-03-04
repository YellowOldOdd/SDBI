#!/usr/bin/env python

import argparse
import numpy as np
import multiprocessing as mp
import threading
import queue
import torch
import os
from datetime import datetime, timedelta
from time import time, sleep

from SimpleDBI.session import Session, Run
from SimpleDBI.backend_metric import BackendMetric

modelfile = 'test.jit'

def time_torch(params) : 
    # 1. create torch model
    device = 'cpu' 
    if torch.cuda.device_count() > 0 :
        device = 'cuda:0'
    model = torch.jit.load(modelfile).to(device).eval()

    # 2. init data
    latency = []
    np_arr = np.random.random(
        size = (params['worker_batch'],3,224,224)).astype(np.float32)
    beg = params['start_timestamp']
    end = params['end_timestamp']

    # 3. performance testing loop
    while time() < end + 1 :
        request_beg = time()
        tensor = torch.from_numpy(np_arr).to(device)
        with torch.no_grad():
            output = model.forward(tensor)
        output = output.detach().cpu().numpy()
        request_end = time()
        # record metric in specific time range
        if request_beg >= beg and request_beg < end :
            latency.append(request_end - request_beg)

    # 4. send metric 
    q = params['latency_queue']
    q.put(latency)

    return

def time_simple_dbi(params) : 
    max_batch_size = params['max_batch_size']

    # 1. create session
    sess = Session(
        name            = 'test', 
        path            = modelfile, 
        model_type      = params['model_type'],
        dynamic_batch   = params['dynamic_batch'],
        duplicate_num   = params['duplicate_num'],
        max_batch_size  = max_batch_size,
        timeout         = params['timeout'],
        metric          = True, 
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

    # 2. init data
    latency = []
    metric_l = []
    beg = params['start_timestamp']
    end = params['end_timestamp']
    np_arr = np.random.random(
        size = (params['worker_batch'],3,224,224)).astype(np.float32)

    # 3. performance testing loop
    while time() < end + 1 :
        request_beg = time()
        output, metric = sess.forward(np_arr)
        request_end = time()
        # record metric in specific time range
        if request_beg >= beg and request_beg < end :
            latency.append(request_end - request_beg)
            metric_l.append(metric)

    # check result by MockModel(copy input to output)
    if params['model_type'] == 'mock' :
        for idx in range(params['worker_batch']) :
            assert output[idx][0] == np_arr[idx][0][0][0]
            assert output[idx][1] == np_arr[idx][0][0][1]
    
    # 4. send metric 
    metric_d = {}
    for k in BackendMetric.keys() :
        metric_d[k] = []

    for m in metric_l :
        m_d = BackendMetric.analysis(m)
        for k, v in m_d.items() :
            metric_d[k].append(v)

    q = params['latency_queue']
    q.put(latency)
    q.put(metric_d)
    return 

def collector(q, batch_size, duration) :
    # session(client) metric
    latency = []
    
    # backend metric
    backend_metric = {}
    for k in BackendMetric.keys() :
        backend_metric[k] = []

    # 1. collecting metric
    while True :
        try :
            l = q.get()
            if l == 'END' :
                break
            if isinstance(l, list) :
                latency.extend(l)
            if isinstance(l, dict) :
                for k, v in l.items() :
                    backend_metric[k].extend(v)
        except queue.Empty :
            break

    # 2. report session(client) metric
    print('------------------------------')
    latency.sort()
    req_num = len(latency)
    print('Session metric :')
    print('avg latency    : {:.4f} ms'.format(1000*sum(latency)/req_num))
    print('avg throughput : {:.4f}'.format(batch_size*req_num/duration))
    print('p50 latency    : {:.4f} ms'.format(1000*latency[int(req_num*0.50)]))
    print('p99 latency    : {:.4f} ms'.format(1000*latency[int(req_num*0.99)]))
    print()
    
    if len(backend_metric) == 0 :
        return
    
    # 3. report backend(server) metric 
    print('----------')
    print('Backend metric :')
    for k,v in backend_metric.items() :
        if k == 'batch_size' :
            continue
        print('avg {:25s}    : {:.4f} ms'.format(k, 1000*sum(v)/len(v)))
    
    batch_sizes = {}
    for v in backend_metric['batch_size'] :
        if batch_sizes.get(v) is None :
            batch_sizes[v] = 0
        batch_sizes[v] += 1
    
    batch_num = 0
    batch_sum = 0
    for bs, num in batch_sizes.items() :
        batch_num += num / bs
        batch_sum += num
        batch_sizes[bs] = num / bs
    print('Batch size histogram : {}'.format(batch_sizes))
    print('Avg batch size : {}'.format(batch_sum / batch_num))
    return 
    
def timer(total_time) :
    # tell you time to wait
    for i in range(total_time):
        sleep(1) 
        print('\rAbout {}s left'.format(total_time - i), end='')
    print('\n')
    return 

def main() :
    parser = argparse.ArgumentParser(
        description='Benchmark for simple dynamic batched inference.')

    parser.add_argument('--dynamic_batch',    dest='dynamic_batch', action='store_true')
    parser.add_argument('--no_dynamic_batch', dest='dynamic_batch', action='store_false')
    parser.set_defaults(dynamic_batch=True)
    parser.add_argument('--worker_num',     type=int,   default=64,      help='Data process/thread num.')
    parser.add_argument('--max_batch_size', type=int,   default=32,      help='Max batch size.')
    parser.add_argument('--worker_batch',   type=int,   default=1,       help='Batch size per worker.')
    parser.add_argument('--model_num',      type=int,   default=1,       help='Number of gpu model.')
    parser.add_argument('--duration',       type=int,   default=60,      help='Test time.')
    parser.add_argument('--model_type',     type=str,   default='torch', help='model type.')
    parser.add_argument('--wait_time',      type=float, default=0.01,      help='Wait time(s).')

    args = parser.parse_args()

    print('--------------------------------------------------')
    print('CONFIGS :')
    print('dynamic_batch         : {}'.format(args.dynamic_batch))
    print('worker_num            : {}'.format(args.worker_num))
    print('worker_batch          : {}'.format(args.worker_batch))
    print('max_batch_size        : {}'.format(args.max_batch_size))
    print('test duration         : {} s'.format(args.duration))
    print('model num             : {}'.format(args.model_num))
    print('model_type            : {}'.format(args.model_type))
    print('max batching time     : {} s'.format(args.wait_time))
    print('--------------------------------------------------')

    # 1. parse args
    worker_num    = args.worker_num
    duration      = args.duration
    current_time  = datetime.now()
    start_time    = current_time + timedelta(seconds=int(worker_num/10)+5+2)
    end_time      = start_time + timedelta(seconds=duration)
    
    # collect letency metric(consider this is a tsdb client)
    latency_q = mp.Queue(maxsize=worker_num * 100)

    params = {
        'dynamic_batch'     : args.dynamic_batch,
        'max_batch_size'    : args.max_batch_size,
        'worker_batch'      : args.worker_batch,
        'duplicate_num'     : args.model_num,
        'start_timestamp'   : start_time.timestamp(),
        'end_timestamp'     : end_time.timestamp(),
        'latency_queue'     : latency_q,
        'model_type'        : args.model_type,
        'timeout'           : args.wait_time,
    }

    # 2. run timer
    timer_proc = threading.Thread(
        target=timer, args = (duration+int(worker_num/10)+5, ))
    timer_proc.start()

    # 3. run metric collector(mock tsdb)
    collector_proc = threading.Thread(
        target=collector, args = (latency_q, args.worker_batch, duration))
    collector_proc.start()

    # 4. run inference benchmark 
    if args.dynamic_batch or args.model_num > 1:
        print('Run Benchmark with Simple Dynamic Batching Inference')
        Run(target = time_simple_dbi, worker_num=worker_num, args = (params,))
    else :
        print('Run Benchmark with Torch')
        proc_list = []
        for i in range (worker_num) :
            proc = mp.Process(target=time_torch, args = (params,))
            proc.start()
            proc_list.append(proc)
        for proc in proc_list :
            proc.join()

    # 5. join and exit
    latency_q.put('END')
    collector_proc.join()
    timer_proc.join()

if __name__ == '__main__'  :
    main()
    