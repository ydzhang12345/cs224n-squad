#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on match-LSTM paper
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import pdb

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
import collections
from tensorflow.python.ops import array_ops

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))
_linear = core_rnn_cell_impl._linear
class precell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tanh, reuse=None):
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be deprecated.  Use state_is_tuple=True.", self)

        if input_size is not None:
          logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = array_ops.split(value=state, num_or_size_splits=2, axis=1)
            concat = _linear([inputs, h], 3 * self._num_units, True)

            i, j, f = array_ops.split(value=concat, num_or_size_splits=3, axis=1)
            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state
