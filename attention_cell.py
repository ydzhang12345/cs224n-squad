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

logger = logging.getLogger("hw3.q3.1")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class AttentionCell(tf.contrib.rnn.RNNCell):
    """Wrapper around our GRU cell implementation that allows us to play
    nicely with TensorFlow.
    """
    def __init__(self, question_size, state_size):
        self._question_size = question_size
        self._state_size = state_size

    @property
    def state_size(self):
        return self._state_size

    @property
    def question_size(self):
        return self._question_size

    def __call__(self, question, inputs, state, cosine, scope=None):
        """Updates the state using the previous @state and @inputs.
        Remember the GRU equations are:

        z_t = sigmoid(x_t U_z + h_{t-1} W_z + b_z)
        r_t = sigmoid(x_t U_r + h_{t-1} W_r + b_r)
        o_t = tanh(x_t U_o + r_t * h_{t-1} W_o + b_o)
        h_t = z_t * h_{t-1} + (1 - z_t) * o_t

        TODO: In the code below, implement an GRU cell using @inputs
        (x_t above) and the state (h_{t-1} above).
            - Define W_r, U_r, b_r, W_z, U_z, b_z and W_o, U_o, b_o to
              be variables of the apporiate shape using the
              `tf.get_variable' functions.
            - Compute z, r, o and @new_state (h_t) defined above
        Tips:
            - Remember to initialize your matrices using the xavier
              initialization as before.
        Args:
            inputs: is the input vector of size [None, self.input_size]
            state: is the previous state vector of size [None, self.state_size]
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        # It's always a good idea to scope variables in functions lest they
        # be defined elsewhere!
        with tf.variable_scope(scope):
            ### YOUR CODE HERE (~20-30 lines)
            '''
            W_q= tf.get_variable("W_q", shape=[self.state_size, self.state_size],
                        initializer=tf.contrib.layers.xavier_initializer())
            W_p= tf.get_variable("W_p", shape=[self.state_size, self.state_size],
                        initializer=tf.contrib.layers.xavier_initializer())
            W_r= tf.get_variable("W_r", shape=[self.state_size, self.state_size],
                        initializer=tf.contrib.layers.xavier_initializer())
            b_p = tf.get_variable("b_p", shape=[self.state_size], initializer=tf.constant_initializer(0.0))

            w = tf.get_variable("w", shape=[self.state_size, 1],
                        initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape=[1], initializer=tf.constant_initializer(0.0))
            '''

            W_q= tf.get_variable("W_q", shape=[self.state_size, self.state_size], initializer=tf.orthogonal_initializer())
            W_p= tf.get_variable("W_p", shape=[self.state_size, self.state_size], initializer=tf.orthogonal_initializer())
            W_r= tf.get_variable("W_r", shape=[self.state_size, self.state_size], initializer=tf.orthogonal_initializer())
            b_p = tf.get_variable("b_p", shape=[self.state_size], initializer=tf.constant_initializer(0.0))

            w = tf.get_variable("w", shape=[self.state_size, 1], initializer=tf.orthogonal_initializer())
            b = tf.get_variable("b", shape=[1], initializer=tf.constant_initializer(0.0))

            # temp is of (batch, 1, state_size)
            temp = tf.expand_dims(
                tf.matmul(inputs, W_p) + tf.matmul(state, W_r) + b_p, axis=1)
            # temp is of (bacth, Q, state)
            temp = tf.tile(temp, tf.stack([1, self.question_size, 1]))

            #temp_b = tf.transpose(tf.tile(
            #    tf.expand_dims(b, 1), tf.stack([1, self.question_size])))
            # G is (batch_size, Q, hidden_state)
            # G = tf.tanh(tf.matmul(question, W_q) + temp) 
            # tensorflow does not support 3D tensor matmul
            # question is (batch_size. Q, hidden_state)

            # op to get question x W_q:
            ques_op = tf.reshape(question, [-1, self.state_size])
            ques_op = tf.matmul(ques_op, W_q)
            ques_op = tf.reshape(ques_op, [-1, self.question_size, self.state_size])
            G = tf.tanh(ques_op + temp)
            G = tf.reshape(G, [-1, self.state_size])
            G_op = tf.reshape(tf.matmul(G, w), [-1, self.question_size])  # -1 at the last dim?
            # G = tf.tanh(tf.einsum('ijk,kl->ijl', question, W_q) + temp)
            # alpha is (batch_size, Q)
            # alpha = tf.nn.softmax(tf.matmul(G, w) + temp_b)
            # tt = tf.reduce_sum(tf.einsum('ijk,kl->ijl', G, w), axis=2)

            # cosine (b, q)
            cos = tf.get_variable("cos", shape=[1], initializer=tf.constant_initializer(0.5))
            alpha = tf.nn.softmax(G_op + cos * cosine + b)
        return alpha
