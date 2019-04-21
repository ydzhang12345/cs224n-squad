from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
import numpy as np
import pdb
from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 6.6, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.5, "Fraction of units randomly kept on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 20, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 15, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 256, "The output size of your model.") #766
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adamax", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_integer("max_length", 20, "max_lenth of the question.") 


FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def load_and_pad_val_data(q_path, c_path, a_path):
    fq, fc, fa = open(q_path), open(c_path), open(a_path)
    ans = []
    question = []
    ques_mask = []
    context = []
    context_mask = []
    mask_c_vec = []

    for _ in range(4284):
        q, c, a = fq.readline(), fc.readline(), fa.readline()
        q = map(int, q.strip().split(' '))
        c = map(int, c.strip().split(' '))
        a = map(int, a.strip().split(' '))
        if (a[0] > a[1]): continue
        if (max(a) >= FLAGS.output_size) or (len(q) > FLAGS.max_length):
            continue
        ques_mask.append(len(q))
        if len(c) > FLAGS.output_size:
            c = c[:FLAGS.output_size]
        context_mask.append(len(c))
        vec = []
        vec.extend([True] * len(c))
        ans.append(a)
        if len(q) < FLAGS.max_length:
            q.extend([0] * (FLAGS.max_length - len(q)))
        question.append(q)
        if len(c) < FLAGS.output_size:
            vec.extend([False] * (FLAGS.output_size - len(c)))
            c.extend([0] * (FLAGS.output_size - len(c)))
        context.append(c)
        mask_c_vec.append(vec)
    return (question, ques_mask, context, context_mask, ans, len(ans), mask_c_vec)

def batch_generator(question_path, context_path, answer_path, batch_size, max_length_q, max_length_c):
    question_info = open(question_path)
    context_info = open(context_path)
    answer_info = open(answer_path)
    q_batch = []
    mask_q = []
    ans = []
    c_batch = []
    mask_c = []
    mask_c_vec = []
    size_of_dataset = 81386
    for __ in range(size_of_dataset):
        q, c, a = question_info.readline(), context_info.readline(), answer_info.readline()
        q = map(int, q.strip().split(' '))
        c = map(int, c.strip().split(' '))
        a = map(int, a.strip().split(' '))
        if (a[0] > a[1]): continue
        if (max(a) >= FLAGS.output_size) or (len(q) > FLAGS.max_length):
            continue
        mask_q.append(len(q))
        if len(c) > FLAGS.output_size:
            c = c[:FLAGS.output_size]
        mask_c.append(len(c))
        vec = []
        vec.extend([True] * len(c))
        ans.append(a)
        if len(q) < max_length_q:
            q.extend([0] * (max_length_q - len(q)))
        q_batch.append(q)
        if len(c) < max_length_c:
            vec.extend([False] * (max_length_c - len(c)))
            c.extend([0] * (max_length_c - len(c)))
        c_batch.append(c)
        mask_c_vec.append(vec)
        if len(ans) == batch_size:
            batch = (q_batch, mask_q, c_batch, mask_c, ans, mask_c_vec)
            yield batch
            q_batch = []
            mask_q = []
            ans = []
            c_batch = []
            mask_c = []
            mask_c_vec = []

def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    # load all in once, maybe better to try batch by batch
    question_path = "./data/squad/train.ids.question"
    context_path = "./data/squad/train.ids.context"
    answer_path = "./data/squad/train.span"

    val_q = "./data/squad/val.ids.question"
    val_c = "./data/squad/val.ids.context"
    val_a = "./data/squad/val.span"
    

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")

    # embeddings is a matrix of shape [vocab_size, embedding_size]
    embeddings = np.load(embed_path)['glove'].astype(np.float32)
    val_data = load_and_pad_val_data(val_q, val_c, val_a)

    # vocab is the mapping from word -> token id
    # rev_vocab is the reverse mapping, from id -> word
    vocab, rev_vocab = initialize_vocab(vocab_path)

    # someone posted that the max length of question is 766
    info = (question_path, context_path, answer_path, FLAGS.batch_size, FLAGS.max_length, FLAGS.output_size)
    '''   
    batch_gen = batch_generator(question_path, context_path, answer_path, FLAGS.batch_size, FLAGS.max_length, FLAGS.output_size)
    i = 0;
    while True:
        batch_gen.next()
        i += 1
        logging.info(i)
    '''

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder, FLAGS, embeddings)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, batch_generator, info, save_train_dir, val_data, rev_vocab)
        #qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)
        
if __name__ == "__main__":
    tf.app.run()
