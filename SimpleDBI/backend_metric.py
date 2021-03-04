#!/usr/bin/env python

class BackendMetric(object) :
    arrive = 0
    input_queue_put = 0 
    input_queue_get = 0
    model_queue_put = 0
    model_queue_get = 0
    output_queue_put = 0
    output_queue_get = 0
    backend_end = 0
    send = 0
    batch_size = 0
    concat = 0

    def keys() :
        return [
            'input_request_handle'  ,
            'input_queue_wait'      ,
            'batching'              ,
            'batch_queue_wait'      ,
            'model'                 ,
            'output_queue_wait'     ,
            'split_output'          ,
            'output_request_handle' ,
            'concat'                ,
            'batch_size'            ,
        ]

    @staticmethod
    def analysis(metric) :
        return {
            'input_request_handle'  : metric.input_queue_put  - metric.arrive,
            'input_queue_wait'      : metric.input_queue_get  - metric.input_queue_put,
            'batching'              : metric.model_queue_put  - metric.input_queue_get,
            'batch_queue_wait'      : metric.model_queue_get  - metric.model_queue_put,
            'model'                 : metric.output_queue_put - metric.model_queue_get,
            'output_queue_wait'     : metric.output_queue_get - metric.output_queue_put,
            'split_output'          : metric.backend_end      - metric.output_queue_get,
            'output_request_handle' : metric.send             - metric.backend_end,
            'concat'                : metric.concat,
            'batch_size'            : metric.batch_size,
        }

def get_lat(name, beg, end) :
    latency = (beg - end) * 1000
    print("{:30s} : {:.5f}".format(name, latency))

def print_metric(metric) :
    get_lat('input_request_handle',  metric.input_queue_put,  metric.arrive)
    get_lat('input_queue_wait',      metric.input_queue_get,  metric.input_queue_put)
    get_lat('batching',              metric.model_queue_put,  metric.input_queue_get)
    get_lat('batch_queue_wait',      metric.model_queue_get,  metric.model_queue_put)
    get_lat('model',                 metric.output_queue_put, metric.model_queue_get)
    get_lat('output_queue_wait',     metric.output_queue_get, metric.output_queue_put)
    get_lat('split_output',          metric.backend_end,      metric.output_queue_get)
    get_lat('output_request_handle', metric.send,             metric.backend_end)

