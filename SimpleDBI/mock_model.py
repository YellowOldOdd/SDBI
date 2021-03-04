#!/usr/bin/env python

import numpy as np

class MockModel() :
    def __init__(self, *args) :
        pass

    def forward(self, *args) :
        result = []
        for in_ts in args :
            # print('-------- > batch size : {}'.format(in_ts.shape))
            batch_size = in_ts.shape[0]
            out_ts = np.random.random(size = [batch_size, 2]).astype(np.float32)
            for b in range(batch_size) :
                out_ts[b][0] = in_ts[b][0][0][0]
                out_ts[b][1] = in_ts[b][0][0][1]
            result.append(out_ts)
        return result