#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from nltk.tokenize import TweetTokenizer
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics

def load_params():

    # Parameters
    # ==================================================

    #open configuration file to load params
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Data loading params
    tf.flags.DEFINE_string("train_path", cfg['data_params']['train_path'], "Data source")
    tf.flags.DEFINE_string("dev_path", cfg['data_params']['dev_path'], "Data source")
    tf.flags.DEFINE_string("test_path", cfg['data_params']['test_path'], "Data source")
    tf.flags.DEFINE_string("embed_file_path", cfg['data_params']['embed_file_path'], "embeddings file path")
    tf.flags.DEFINE_string("x_header", cfg['data_params']['x_header'], "x header name")
    tf.flags.DEFINE_string("y_header", cfg['data_params']['y_header'], "y header name")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", cfg['hyper_params']['embedding_dim'], "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", cfg['hyper_params']['filter_sizes'], "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", cfg['hyper_params']['num_filters'], "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", cfg['hyper_params']['dropout_keep_prob'], "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", cfg['hyper_params']['l2_reg_lambda'], "L2 regularization lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", cfg['train_params']['batch_size'], "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", cfg['train_params']['num_epochs'], "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", cfg['train_params']['evaluate_every'], "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", cfg['train_params']['checkpoint_every'], "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", cfg['train_params']['num_checkpoints'], "Number of checkpoints to store (default: 5)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", cfg['misc_params']['allow_soft_placement'], "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", cfg['misc_params']['log_device_placement'], "Log placement of ops on devices")
    tf.flags.DEFINE_integer("random_state", cfg['misc_params']['random_state'], "random seed")

    global FLAGS
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")


def load_data():

    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.data_file)

    # Build vocabulary
    tokenizer = TweetTokenizer(reduce_len=True, strip_handles=True)
    max_document_length = max([len(tokenizer.tokenize(x)) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    '''
    ## Extract word:id mapping from the object.
    vocab_dict = vocab_processor.vocabulary_._mapping

    ## Sort the vocabulary dictionary on the basis of values(id).
    ## Both statements perform same task.
    #sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
    sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])

    ## Treat the id's as index into list and create a list of words in the ascending order of id's
    ## word with id i goes at index i of the list.
    vocabulary1 = list(list(zip(*sorted_vocab))[0])

    #print(vocab_dict)
    #print(x) '''

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    print(shuffle_indices)
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print("Negative = "+str(sum([val[0] for val in y_dev])))
    print("Positive="+str(sum([val[1] for val in y_dev])))

def build_vocabulary(x_text):
    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    global vocab_processor
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    return x

def train_CNN(x_train, y_train, x_dev, y_dev):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy and f1_Score
            f1score = tf.Variable(0, name="f1score", trainable=False)
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            f1score_summary = tf.summary.scalar("f1_score", f1score)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            #vocabulary = vocab_processor.vocabulary_
            #initW = data_helpers.load_embedding_vectors_word2vec(vocabulary, FLAGS.embed_file_path)
            #sess.run(cnn.W.assign(initW))

            dev_final_f1score = None

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, predictions, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.predictions, cnn.accuracy],
                    feed_dict)
                train_f1 = metrics.f1_score(np.argmax(y_batch, axis=1), predictions, average='weighted')
                nonlocal f1score
                f1score = train_f1
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, fscore {:g}".format(time_str, step, loss, accuracy, train_f1))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, predictions = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                dev_f1score = metrics.f1_score(np.argmax(y_batch, axis=1), predictions, average='weighted')
                nonlocal f1score
                f1score = dev_f1score
                nonlocal dev_final_f1score
                print("Ev{}: step {}, loss {:g}, acc {:g}, fscore {:g}".format(time_str, step, loss, accuracy, dev_f1score))
                if writer: #####(dev_final_f1score is None or dev_f1score > dev_final_f1score) and
                    writer.add_summary(summaries, step)
                    dev_final_f1score = dev_f1score

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
            return dev_final_f1score

if __name__ == "__main__":
    #load params from config file
    load_params()
    global vocab_processor
    x_text, y_train = data_helpers.load_data_and_labels(FLAGS.train_path)
    x_train = build_vocabulary(x_text)
    x_dev_text, y_dev = data_helpers.load_data_and_labels(FLAGS.dev_path)
    x_dev = np.array(list(vocab_processor.transform(x_dev_text)))
    f1 = train_CNN(x_train, y_train, x_dev, y_dev)
