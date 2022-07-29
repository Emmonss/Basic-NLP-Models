
import os,sys,six,re,json
import logging
import numpy as np
from collections import defaultdict
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras as keras

_open_ = open
is_py2 = six.PY2

if not is_py2:
    basestring = str

def is_string(item):
    return isinstance(item,basestring)

def truncate_sequences(max_len,indices,*sequences):
    '''
    截断总长度至不超过maxlen
    :param max_len:
    :param indices:
    :param sequences:
    :return:
    '''
    sequences = [s for s in sequences if s]
    if not isinstance(indices,(list,tuple)):
        indices = [indices] * len(sequences)

    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) >max_len:
            i = np.argmax(lengths)
            sequences[i].pop(indices[i])
        else:
            return sequences