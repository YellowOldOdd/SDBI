#!/usr/bin/env python

import logging
import multiprocessing as mp

from SimpleDBI import EXIT_SIG
from SimpleDBI.backend import Backend, backend_process

logger = logging.getLogger('context')

def context_server(ctx_q, metric_q) :
    backend_entry = {}     # comm queue dict
    backend_processes = []  # 
    logger.debug('Context_server started ...')

    while True :
        value = ctx_q.get()
        if value == EXIT_SIG :
            break
        
        conn, args = value
        entry = backend_entry.get(args['name'])
        if entry is None :    
            # 1. create entry queue
            entry = mp.Queue()
            
            # 2. create backend
            process = mp.Process(target=backend_process, args=(entry, metric_q, args))
            # process.daemon = True
            process.start()
            backend_processes.append(process)
            
            # 3. registe 
            backend_entry[args['name']] = entry
        
        entry.put(conn)

    for k, v in backend_entry.items() :
        v.put(EXIT_SIG)

    for proc in backend_processes :
        proc.join()

    for k, v in backend_entry.items() :
        v.close()

    logger.debug('Context Server exit.')
