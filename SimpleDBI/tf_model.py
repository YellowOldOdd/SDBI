#!/usr/bin/env python
# coding: utf-8

"""
The TFModel only suit to TensorFlow 1.x
"""

import copy
import logging 
import numpy as np
import threading

import tensorflow as tf 
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

from SimpleDBI.model import Model

logger = logging.getLogger('tf_model')
logger.setLevel(logging.ERROR)

def import_graph_def_to_graph(graph_def_path):
    """Load a frozen graph(.pb) to tf.Graph()."""
    infer_graph_def = tf_compat_v1.GraphDef()
    with tf_compat_v1.gfile.GFile(graph_def_path, 'rb') as f:
        infer_graph_def.ParseFromString(f.read())
    g = tf_compat_v1.Graph()
    with g.as_default():
        tf_compat_v1.import_graph_def(infer_graph_def, name='')
    return g

class TFModel(object) :
  def __init__(self, model_name, model_path, input_info, output_info) : 
    self.name = model_name
    self.graph = import_graph_def_to_graph(model_path)
    self.input_info = input_info
    self.output_info = output_info
    self.device = "CPU:0"
    self.config = None
    if tf_compat_v1.test.is_gpu_available() :
        self.device = "GPU:0"
        self.config = tf_compat_v1.ConfigProto(device_count={'GPU': 0})
    logger.warning('TensorFlow device {}.'.format(self.device))

  def get_input(self, args) :
    if isinstance(args, tuple) :
        input_array = args[0]
    elif isinstance(args, np.ndarray) :
        input_array = args
    # NCHW -> NHWC
    transposed_array = copy.deepcopy(np.transpose(input_array, (0, 2, 3 ,1)))
    return transposed_array

  def get_graph_node(self, tensor_info):
    """
    When we create a shared_memory.SharedMemory, we use input/output node name as 
    an argument to the SharedMemory's constructor `name` parameter. Unfortunately, 
    the format of tensorflow's node_name is `namescope_1/namescope2/node_name`, 
    which has illegal slash in node_name. 

    Therefore, when we create a input_info/output_info, if a node_name has slashs, 
    we change all slashs to hyphens. The changed node_name is `namescope_1-namescope2-node_name`.
    See tf_sample.py as an example.

    In this function, we change back node_name to the original format.
    """
    node_name = tensor_info[0].name
    if "-" in node_name:
      name_list = node_name.split("-")
      node_name = "/".join(name_list)
    node = self.graph.get_tensor_by_name(node_name + ":0")
    return node      

  def model_forward(self, args) :
    input_node = self.get_graph_node(self.input_info)
    output_node = self.get_graph_node(self.output_info)
    input_tensor = self.get_input(args)
    with tf_compat_v1.Session(graph=self.graph, config=self.config) as sess:
        output = sess.run(output_node, feed_dict={input_node: input_tensor})
    return output

  def forward(self, *args) :
    output = self.model_forward(args)
    return (output, )
