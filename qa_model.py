from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
from util import Progbar, minibatches
import pdb
import math
import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from attention_cell import AttentionCell
from evaluate import exact_match_score, f1_score
from tensorflow.contrib.rnn import DropoutWrapper
from pre_cell import precell
from Adamax import AdamaxOptimizer

logging.basicConfig(level=logging.INFO)


class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, inputs, masks, encoder_state_input, rate):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input:(Optional) pass this as initial hidden state
                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
        It can be context-level representation, word-level representation,
                 or both.
        """
        cell = precell(
            num_units=self.size, state_is_tuple=True)
        d_cell = DropoutWrapper(cell, input_keep_prob=rate)
        initial_fw = None
        initial_bw = None
        if encoder_state_input:
            (initial_fw, initial_bw) = encoder_state_input
        outputs, final = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=d_cell,
            cell_bw=d_cell,
            initial_state_fw=initial_fw,
            initial_state_bw=initial_bw,
            dtype=tf.float32,
            sequence_length=masks,
            inputs=inputs)

        # outputs is a tuple (batch_size, time_steps, 2*hidden_size) 
        outputs = tf.concat(outputs, axis=2)
        return outputs, 2 * self.size, final


def mul_3x2(tensor1, tensor2, dim1, dim2, dim3):
    # tensor1: (batch, dim1, dim2)
    # tensor2: (dim2, dim3)
    t1 = tf.reshape(tensor1, [-1, dim2])
    t1 = tf.matmul(t1, tensor2)
    t1 = tf.reshape(t1, [-1, dim1, dim3])
    return t1


def mul_2x3(tensor1, tensor2, dim1, dim2):
    # tensor1: (batch, dim1)
    # tensor2: (batch, dim1, dim2)
    t1 = tf.expand_dims(tensor1, axis=1)
    t1 = tf.matmul(t1, tensor2)
    t1 = tf.reduce_sum(t1, axis=1)
    return t1

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, H_match, info_size, state_size, hidden_shape, r_info, rate):

        # Answer Pointer Layer
        lstmcell = tf.contrib.rnn.BasicLSTMCell(
            state_size, state_is_tuple=True)
        d_cell = DropoutWrapper(
            lstmcell, input_keep_prob=rate)

        with tf.variable_scope("Answer_Pointer"):
            c_a_lstm = tf.zeros(hidden_shape, tf.float32)
            h_a_lstm = tf.zeros(hidden_shape, tf.float32)
            V = tf.get_variable("V", shape=[info_size, state_size], initializer=tf.orthogonal_initializer())
            v = tf.get_variable("v", shape=[state_size, 1], initializer=tf.orthogonal_initializer())
            b_a = tf.get_variable(
                "b_a", shape=[state_size],
                initializer=tf.constant_initializer(0.0))
            c = tf.get_variable(
                "c", shape=[1], initializer=tf.constant_initializer(0.0))
            W_a = tf.get_variable("W_a", shape=[state_size, state_size], initializer=tf.orthogonal_initializer())
            W_re = tf.get_variable("W_re", shape=[info_size, state_size], initializer=tf.orthogonal_initializer())

            # F_s of shape (batch, context_size, state_size)

            temp = tf.expand_dims(
                tf.matmul(h_a_lstm, W_a) + tf.matmul(r_info, W_re), axis=1)
            # note output_size refers to the paragraph size here
            temp = tf.tile(temp, tf.stack([1, self.output_size, 1]))
            F_s = tf.tanh(mul_3x2(H_match, V, self.output_size, info_size, state_size) + temp + b_a)
            # F_s = tf.einsum("ijk,kl->ijl", H_match, V) + temp + b_a

            # temp = tf.reduce_sum(tf.einsum("ijk,kp->ijp", F_s, v), axis=2)
            temp = tf.reduce_sum(mul_3x2(F_s, v, self.output_size, state_size, 1), axis=2)
            s_logit = temp + c
            beta_s = tf.nn.softmax(s_logit)
            # lstm_input = tf.einsum("ij,ijk->ik", beta_s, H_match)
            lstm_input = mul_2x3(beta_s, H_match, self.output_size, info_size)
            __, (c_a_lstm, h_a_lstm) = d_cell(
                lstm_input, (c_a_lstm, h_a_lstm))

            # for the end indice: maybe can do a bi-direction here. 
            temp_e = tf.expand_dims(
                tf.matmul(h_a_lstm, W_a) + tf.matmul(r_info, W_re), axis=1)
            # note output_size refers to the paragraph size here
            temp_e = tf.tile(temp_e, tf.stack([1, self.output_size, 1]))
            # F_e = tf.einsum("ijk,kl->ijl", H_match, V) + temp_e + b_a
            F_e = tf.tanh(mul_3x2(H_match, V, self.output_size, info_size, state_size) + temp_e + b_a)
            # temp_e = tf.reduce_sum(tf.einsum("ijk,kp->ijp", F_e, v), axis=2)
            temp_e = tf.reduce_sum(mul_3x2(F_e, v, self.output_size, state_size, 1), axis=2)
            e_logit = temp_e + c
            #beta_e = tf.nn.softmax(e_logit)
            #confidence = tf.sigmoid(mul_2x3(beta_s, H_match, self.output_size, info_size) + mul_2x3(beta_e, H_match, self.output_size, info_size))

            
        return s_logit, e_logit

class QASystem(object):
    def __init__(self, encoder, decoder, FLAGS, pretrained_embeddings, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.config = FLAGS
        self.pretrained_embeddings = pretrained_embeddings
        self.global_step = tf.get_variable("global", initializer=tf.constant(1), trainable=False)
        
        # ==== set up placeholder tokens ========
        self.question_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, self.config.max_length])
        self.question_mask = tf.placeholder(tf.int32, shape=[None])

        self.context_placeholder = tf.placeholder(
            tf.int32,
            shape=[None, self.config.output_size])
        self.context_mask = tf.placeholder(tf.int32, shape=[None])

        self.ans_placeholder = tf.placeholder(tf.int32, shape=[None, 2])
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.mask_placeholder = tf.placeholder(tf.bool, shape=[None, self.config.output_size])

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.orthogonal_initializer()):
            question, context, cosine = self.setup_embeddings()
            pred_s, pred_e = self.setup_system(question, context, cosine)
            self.loss, self.EM = self.setup_loss(pred_s, pred_e)

        # ==== set up training/updating procedure ====
        #self.train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        optimizer = AdamaxOptimizer(self.config.learning_rate)
        gradients = optimizer.compute_gradients(self.loss)
        norm_list = []
        clipped = []
        for gv in gradients:
            norm_list.append(gv[0])
        #if self.config.clip_gradients is True:
        norm_list, global_norm = tf.clip_by_global_norm(norm_list, self.config.max_gradient_norm)
        self.grad_norm = tf.global_norm(norm_list)
        for i, gv in enumerate(gradients):
            clipped.append((norm_list[i], gv[1]))
        self.train_op = optimizer.apply_gradients(clipped, global_step=self.global_step)
       
    def setup_system(self, question, context, cosine):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        #question, context = self.setup_embeddings()
        with tf.variable_scope("question_encoder"):
            self.question_states, self.new_hidden_size, q_info = self.encoder.encode(
                inputs=question,
                masks=self.question_mask,
                rate=self.dropout_placeholder,
                encoder_state_input=None)
            tf.get_variable_scope().reuse_variables()
            self.context_states, __, __ = self.encoder.encode(
                inputs=context,
                masks=self.context_mask,
                rate=self.dropout_placeholder,
                encoder_state_input=None)

        # attention mechannism based on match-LSTM layer
        # hidden_shape is (batch, hidden)
        hidden_shape = tf.shape(self.context_states[:, 0, :])
        h_fw = tf.zeros(hidden_shape, tf.float32)
        c_fw = tf.zeros(hidden_shape, tf.float32)
        fw_list = []
        bw_list = []
        h_bw = tf.zeros(hidden_shape, tf.float32)
        c_bw = tf.zeros(hidden_shape, tf.float32)
        cell = AttentionCell(
            self.config.max_length, self.new_hidden_size)
        lstmcell = tf.contrib.rnn.BasicLSTMCell(
            self.new_hidden_size, state_is_tuple=True)
        # d_cell = DropoutWrapper(
        #     lstmcell, input_keep_prob=self.dropout_placeholder)
        with tf.variable_scope("Match-LSTM_q2c"):         
            for time_step in range(self.config.output_size):                  
                with tf.variable_scope("alpha"):
                    alpha_fw = cell(
                        self.question_states,
                        self.context_states[:, time_step, :], h_fw, cosine[:, :, time_step])
                    tf.get_variable_scope().reuse_variables()
                    alpha_bw = cell(
                        self.question_states,
                        self.context_states[:, self.config.output_size - time_step - 1, :], h_bw, cosine[:, :, self.config.output_size - time_step - 1])
                # alpha is (batch_size, Q), temp is (batch, hidden_state)
                # question_states is (batch, Q, hidden)
                temp_fw = mul_2x3(alpha_fw, self.question_states, self.config.max_length, self.new_hidden_size)
                z_fw = tf.concat([self.context_states[:, time_step, :], temp_fw], axis=1)
                __, (c_fw, h_fw) = lstmcell(z_fw, (c_fw, h_fw))
                tf.get_variable_scope().reuse_variables() 
                fw_list.append(tf.expand_dims(h_fw, axis=1))
                temp_bw = mul_2x3(alpha_bw, self.question_states, self.config.max_length, self.new_hidden_size)
                z_bw = tf.concat([self.context_states[:, self.config.output_size - time_step - 1, :], temp_bw], axis=1)
                __, (c_bw, h_bw) = lstmcell(z_bw, (c_bw, h_bw))
                bw_list.append(tf.expand_dims(h_bw, axis=1))
        # now you need to concatenate the two list,
        # and get a new H_match of shape (batch, P, 2*new_states_size)
        H_fw = tf.concat(fw_list, axis=1)
        H_bw = tf.concat(bw_list[::-1], axis=1)
        H_match = tf.concat([H_fw, H_bw], axis=2)
        
        
        cell_c2q = AttentionCell(
            self.config.output_size, self.new_hidden_size)
        hh_fw = tf.zeros(hidden_shape, tf.float32)
        cc_fw = tf.zeros(hidden_shape, tf.float32)
        hh_bw = tf.zeros(hidden_shape, tf.float32)
        cc_bw = tf.zeros(hidden_shape, tf.float32)
        with tf.variable_scope("Match-LSTM_c2q"):         
            for time_step in range(self.config.max_length):     
                with tf.variable_scope("alpha"):
                    alpha_fw = cell_c2q(
                        self.context_states,
                        self.question_states[:, time_step, :], hh_fw, cosine[:, time_step, :])
                    tf.get_variable_scope().reuse_variables()
                    alpha_bw = cell_c2q(
                        self.context_states,
                        self.question_states[:, self.config.max_length - time_step - 1, :], hh_bw, cosine[:, self.config.max_length - time_step - 1, :])
                # alpha is (batch_size, Q), temp is (batch, hidden_state)
                # question_states is (batch, Q, hidden)
                temp_fw = mul_2x3(alpha_fw, self.context_states, self.config.output_size, self.new_hidden_size)
                z_fw = tf.concat([self.question_states[:, time_step, :], temp_fw], axis=1)
                __, (cc_fw, hh_fw) = lstmcell(z_fw, (cc_fw, hh_fw))
                tf.get_variable_scope().reuse_variables() 
                temp_bw = mul_2x3(alpha_bw, self.context_states, self.config.output_size, self.new_hidden_size)
                z_bw = tf.concat([self.question_states[:, self.config.max_length - time_step - 1, :], temp_bw], axis=1)
                __, (cc_bw, hh_bw) = lstmcell(z_bw, (cc_bw, hh_bw))
        
        h_match_c2q = tf.concat([hh_fw, hh_bw], axis=1)
        self.match_hidden_size = self.new_hidden_size * 2
        
        # Answer Pointer Layer
        
        start, end = self.decoder.decode(
            H_match, self.match_hidden_size, self.new_hidden_size, hidden_shape, h_match_c2q, 0.5 + self.dropout_placeholder / 2)
        
        #self.match_hidden_size = self.new_hidden_size * 2
        #start, end = self.decoder.decode(
        #    H_match, self.match_hidden_size, self.new_hidden_size, hidden_shape, 0.5 + self.dropout_placeholder / 2)
        return start, end

    def setup_loss(self, pred_s, pred_e):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            # first set up one-hot vector
            # true returns (batch, 2, context_size)
            # true = tf.one_hot(
            #    self.ans_placeholder, self.config.output_size, axis=-1)
            # pdb.set_trace()
            true_s, true_e = tf.unstack(self.ans_placeholder, axis=1)

            pred_s = tf.add(pred_s, (1 - tf.cast(self.mask_placeholder, 'float')) * (-1e30), name="exp_mask_s")
            pred_e = tf.add(pred_e, (1 - tf.cast(self.mask_placeholder, 'float')) * (-1e30), name="exp_mask_e")

            self.ys = tf.nn.softmax(pred_s)
            self.ye = tf.nn.softmax(pred_e)
            a_s = tf.argmax(self.ys, axis=1)
            a_e = tf.argmax(self.ye, axis=1)
            EM_s = tf.cast(tf.equal(tf.to_int32(a_s), true_s), tf.float32)
            EM_e = tf.cast(tf.equal(tf.to_int32(a_e), true_e), tf.float32)
            EM = tf.reduce_mean(tf.multiply(EM_s, tf.transpose(EM_e)))
            
            loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_s, labels=true_s)
            loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_e, labels=true_e)
            loss = tf.reduce_mean(loss_s + loss_e)
              
        return loss, EM
            
    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            embedding_tensor = tf.constant(self.pretrained_embeddings)
            embeddings_q = tf.nn.embedding_lookup(embedding_tensor, self.question_placeholder)
            embeddings_question = tf.reshape(embeddings_q, [-1, self.config.max_length, self.config.embedding_size])
            embeddings_c = tf.nn.embedding_lookup(embedding_tensor, self.context_placeholder)
            embeddings_context = tf.reshape(embeddings_c, [-1, self.config.output_size, self.config.embedding_size])

            # filter layer
            # cosine (b, q, p)
            cosine = tf.matmul(
                tf.nn.l2_normalize(embeddings_question, dim=2),
                tf.transpose(tf.nn.l2_normalize(embeddings_context, dim=2), perm=[0, 2, 1]))

        return embeddings_question, embeddings_context, cosine

    def evaluate_answer(self, session, dataset, rev_vocab, sample=1500):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """
        f1 = 0.
        em = 0.
        q, q_m, c, c_m, a, l, mask = dataset
        idx = random.sample(xrange(l), sample)
        q_batch = [q[i] for i in idx]
        mask_q = [q_m[i] for i in idx]
        c_batch = [c[i] for i in idx]
        mask_c = [c_m[i] for i in idx]
        ans = [a[i] for i in idx]
        masker = [mask[i] for i in idx]
        feed_dict = {self.question_placeholder: q_batch, self.question_mask: mask_q,
                     self.context_placeholder: c_batch, self.context_mask: mask_c,
                     self.ans_placeholder: ans, self.dropout_placeholder: 1.0,
                     self.mask_placeholder: masker
                     }
        #_, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed_dict)
        a_s, a_e = session.run([self.ys, self.ye], feed_dict=feed_dict)
        # perform beam search on the answer, beam width = 5
        for i in range(sample):
            start = a_s[i]
            end = a_e[i]
            ind_s = np.argpartition(start, -10)[-10:]
            ind_e = np.argpartition(end, -10)[-10:]
            max_prob = 0
            max_s = 0
            max_e = 0
            for j in range(10):
                for k in range(10):
                    if (ind_s[j] <= ind_e[k]) and (ind_e[k] <= mask_c[i]) and (ind_e[k] - ind_s[j] < 15):
                        prob = start[ind_s[j]] * end[ind_e[k]]
                        if prob > max_prob:
                            max_prob = prob
                            max_s = ind_s[j]
                            max_e = ind_e[k]

            string = ""
            truth = ""
            for j in range(max_s, max_e + 1):
                string += rev_vocab[j] + " "
            for j in range(ans[i][0], ans[i][1] + 1):
                truth += rev_vocab[j] + " "

            em = em + exact_match_score(str(max_s) + ' ' + str(max_e), str(ans[i][0]) + ' ' + str(ans[i][1]))
            f1 = f1 + f1_score(string, truth)
        f1 = float(f1) / sample
        em = float(em) / sample
        return f1, em

    def train_on_batch(self, sess, batch):
        (q_batch, mask_q, c_batch, mask_c, ans, masker) = batch
        feed_dict = {self.question_placeholder: q_batch, self.question_mask: mask_q,
                     self.context_placeholder: c_batch, self.context_mask: mask_c,
                     self.ans_placeholder: ans, self.dropout_placeholder: self.config.dropout,
                     self.mask_placeholder: masker
                     }

        #_, loss, grad_norm = sess.run([self.train_op, self.loss, self.grad_norm], feed_dict=feed_dict)
        __, loss, EM, grad_norm = sess.run([self.train_op, self.loss, self.EM, self.grad_norm], feed_dict=feed_dict)
        return loss, grad_norm, EM

    def run_epoch(self, sess, batch_gen, info):
        # use 3301 for 24 batch size
        # use 2476 for 32 batch size
        prog = Progbar(target=4952)
        (i1, i2, i3, i4, i5, i6) = info
        batch_epoch = batch_gen(i1, i2, i3, i4, i5, i6)
        for i in range(4952):
            batch = batch_epoch.next()
            loss, grad_norm, EM = self.train_on_batch(sess, batch)
            logging.info("loss is %f, grad_norm is %f" % (loss, grad_norm))
            prog.update(i + 1, [("train loss", loss), ("grad_norm", grad_norm), ("EM", EM)])
            if math.isnan(loss):
                logging.info("loss nan")
                assert False

    def train(self, session, batch_gen, info, train_dir, val_data, rev_vocab):

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
        saver = tf.train.Saver()

        for epoch in range(self.config.epochs):
            self.run_epoch(session, batch_gen, info)
            logging.info("Evaluating on val set:........")
            f1, em = self.evaluate_answer(session, val_data, rev_vocab)
            logging.info("Number of epoch: %d (f1 is %f, EM is %f)" % (epoch + 1, f1, em))
            saver.save(session, 'adamax', global_step=epoch) 
