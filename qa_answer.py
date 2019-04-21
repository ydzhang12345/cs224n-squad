from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin
import pdb

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data

import logging

logging.basicConfig(level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float("learning_rate", 0.002, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.6, "Fraction of units randomly kept on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 15, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 100, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("output_size", 256, "The output size of your model.")
tf.app.flags.DEFINE_integer("max_length", 20, "max_lenth of the question.") 
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")


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


def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []
    context_mask = []
    query_mask = []
    mask = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)
            # note this function is added by ourselves

            if len(context_tokens) > FLAGS.output_size:
                context_tokens = context_tokens[:FLAGS.output_size]
            vec = []
            vec.extend([True] * len(context_tokens))
            if len(context_tokens) < FLAGS.output_size:
                vec.extend([False] * (FLAGS.output_size - len(context_tokens)))

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                # note this part
                if len(question_tokens) > FLAGS.max_length:
                    question_tokens = question_tokens[:FLAGS.max_length]
                query_mask.append(len(question_tokens))
                question_uuid = qas[qid]['id']

                context_ids = [vocab.get(w, qa_data.UNK_ID) for w in context_tokens]
                context_mask.append(len(context_ids))
                if len(context_ids) < FLAGS.output_size:
                    context_ids.extend([0] * (FLAGS.output_size - len(context_ids)))
                qustion_ids = [vocab.get(w, qa_data.UNK_ID) for w in question_tokens]
                if len(qustion_ids) < FLAGS.max_length:
                    qustion_ids.extend([0] * (FLAGS.max_length - len(qustion_ids)))

                # context_data.append(' '.join(context_ids))
                # query_data.append(' '.join(qustion_ids))
                context_data.append(context_ids)
                query_data.append(qustion_ids)
                question_uuid_data.append(question_uuid)
                mask.append(vec)

    return context_data, query_data, question_uuid_data, context_mask, query_mask, mask


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data, c_m, q_m, mask = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data, c_m, q_m, mask


def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}
    c, q, q_id, c_m, q_m, masker = dataset
    bacth = 1000
    l = 0
    # perform beam search on the answer, beam width = 5
    for b in range(1, int(len(c) / bacth) + 2):
        logging.info("b: %d" % (b))
        ss = (b - 1) * bacth
        ee = b * bacth
        cc = c[ss:ee]
        qq = q[ss:ee]
        cc_m = c_m[ss:ee]
        qq_m = q_m[ss:ee]
        qq_id = q_id[ss:ee]
        mask = masker[ss:ee]

        feed_dict = {model.question_placeholder: qq, 
                     model.question_mask: qq_m,
                     model.context_placeholder: cc, 
                     model.context_mask: cc_m,
                     model.dropout_placeholder: 1.0,
                     model.mask_placeholder: mask
                     }

        a_s, a_e = sess.run([model.ys, model.ye], feed_dict=feed_dict)
        for i in range(len(cc)):
            l += 1
            start = a_s[i]
            end = a_e[i]
            ind_s = np.argpartition(start, -10)[-10:]
            ind_e = np.argpartition(end, -10)[-10:]
            max_prob = 0
            max_s = 0
            max_e = 0
            for j in range(10):
                for k in range(10):
                    if (ind_s[j] <= ind_e[k]) and (ind_e[k] <= cc_m[i]) and (ind_e[k] - ind_s[j] < 15):
                        prob = start[ind_s[j]] * end[ind_e[k]]
                        if prob > max_prob:
                            max_prob = prob
                            max_s = ind_s[j]
                            max_e = ind_e[k]

            string = ""
            for j in range(max_s, max_e + 1):
                p_token_id = cc[i][j]
                string += rev_vocab[p_token_id] + " "

            answers[qq_id[i]] = string

    logging.info("l: %d" % (l))
    return answers


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


def main(_):

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, question_uuid_data, c_m, q_m, mask = prepare_dev(dev_dirname, dev_filename, vocab)
    dataset = (context_data, question_data, question_uuid_data, c_m, q_m, mask)
    embeddings = np.load(embed_path)['glove'].astype(np.float32)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder, FLAGS, embeddings)

    with tf.Session() as sess:
        train_dir = get_normalized_train_dir(FLAGS.train_dir)
        initialize_model(sess, qa, train_dir)
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
