# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pdb

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from pycocoevalcap.eval import COCOScorer
import pickle as pkl
import json
import copy

import data_utils
import seq2seq_model

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.

gConfig = {}

def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    # _conf_booleans = [ (key, bool(value)) for key,value in parser.items('booleans') ]
    _conf_booleans = [ (name, parser.getboolean('booleans', name))
                        for name in parser.options('booleans') ]
    return dict(_conf_ints + _conf_floats + _conf_strings + _conf_booleans)

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(10, 15), (10, 20), (15, 30), (15, 50)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def process_data(source_vectors, source_info, target_info):
    """ Put vectors from the same video description into the same list.

    Args:
    source_vectors: a pickle file containing vector representation of sentences.
    source_info: infomation file containing the sequence id and instance id for corresponding vectors in source vectors/sentences.
    target_info: infomation file containing the sequence id and instance id for corresponding vectors in target vectors/sentences.

    Returns:
    source: a list of vector collections. Each vector collection corresponds to a description with a unique pair of sequence id and instance id.

    for TACoS,
    len(source) = 2862 which is the number of videos;
    len(source[i]) is the number of sentences in description i;
    len(source[i][j]) is 300 which is the dimension of sentence vectors.
    """
    source_seq_id_list = source_info['sequence_id_list']
    source_ins_id_list = source_info['instance_id_list']
    target_seq_id_list = target_info['sequence_id_list']
    target_ins_id_list = target_info['instance_id_list']
    assert (len(target_seq_id_list) == len(target_ins_id_list)),"invalid train_dec_info_file file."

    source = []
    info = []  # a list of list [seq_id, ins_id]
    prev_seq_id, prev_ins_id = None, None
    for i in xrange(len(target_seq_id_list)):
        current_seq_id = target_seq_id_list[i]
        current_ins_id = target_ins_id_list[i]

        if (current_seq_id != prev_seq_id) or (current_ins_id != prev_ins_id):  # This is a different video
            seq_ind = [j for j, x in enumerate(source_seq_id_list) if x == current_seq_id]
            ins_ind = [j for j, x in enumerate(source_ins_id_list) if x == current_ins_id]
            ind = list(set(seq_ind).intersection(ins_ind))
            ind.sort()
            current_video_vectors = [source_vectors[k] for k in ind]
            source.append(current_video_vectors)
            info.append([current_seq_id, current_ins_id])

        prev_seq_id = current_seq_id
        prev_ins_id = current_ins_id

    return source, info


def read_vec_sent(vec_enc, source_info_file, sent_dec, target_info_file, max_size=None):
    """ Read vectors generated from encoder for source.
        Read sentences for target.
        Put them into buckets.

    Args:
    vec_enc: the pkl/np file for the source vectors. Read as list of numpy arrays.
    info_file: infomation file for the vectors in vec_enc. json format.
    sent_dec: path to the file with token-ids for the target language;
        it must be aligned with vec_enc: n-th line contains the desired
        output for n-th line from vec_enc.
    max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of vec/token-ids.
    """
    data_set = [[] for _ in _buckets]
    data_info = [[] for _ in _buckets]
    if gConfig['vector_src'] == 'sent2vec':
        with open(vec_enc, 'r') as source_file:
            source_vectors = pkl.load(source_file)
    elif gConfig['vector_src'] == 'skipthought':
        source_vectors = list(np.load(vec_enc))
    elif gConfig['vector_src'] == 'skipgram':
        with open(vec_enc, 'r') as source_file:
            data = source_file.read().split('\n')
        del data[0], data[-1]
        source_vectors = []
        for i in xrange(len(data)):
            tmp = data[i].split()
            del tmp[0]
            source_vectors.append(np.asarray(tmp, dtype=np.float32))
        del tmp, data
    else:
        raise ValueError("%d cannot be regonized." % gConfig['vector_src'])

    with open(source_info_file, 'r') as source_information_file:
        source_info = json.load(source_information_file)
    with open(target_info_file, 'r') as target_information_file:
        target_info = json.load(target_information_file)
    source, info = process_data(source_vectors, source_info, target_info)
        # Note: for TACoS short-singlesentence task, len(source) = 2862 which is the number of videos; len(source[i]) is the number of sentences in description i; len(source[i][j]) is 300 which is the dimension of sentence vectors.))

    with tf.gfile.GFile(sent_dec, mode="r") as target_file:
        target = target_file.readline()
        counter = 0

        while counter < len(source) and target and (not max_size or counter < max_size):
            # "source[counter]" corresponds to "target".
            current_source = source[counter]
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            target_ids = [int(x) for x in target.split()]
            target_ids.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                if len(current_source) < source_size and len(target_ids) < target_size:
                    data_set[bucket_id].append([current_source, target_ids])
                    data_info[bucket_id].append(info[counter])
                    break
            counter += 1
            target = target_file.readline()

    # Some text file in tacos dateset is empty.
    # We remove the enc-dec pairs with empty entries.
    data_set_copy = copy.deepcopy(data_set)
    data_info_copy = copy.deepcopy(data_info)
    for bucket_id in range(len(data_set_copy)):
        for pair in range(len(data_set_copy[bucket_id])):
            enc = data_set_copy[bucket_id][pair][0]
            dec = data_set_copy[bucket_id][pair][1]
            if len(enc) == 0 or len(dec) == 0:
                data_set[bucket_id].remove(data_set_copy[bucket_id][pair])
                data_info[bucket_id].remove(data_info_copy[bucket_id][pair])
    del data_set_copy, data_info_copy

    return data_set, data_info


def create_model(session, forward_only):

  """Create model and initialize or load parameters"""
  model = seq2seq_model.Seq2SeqModel(gConfig['dec_vocab_size'], _buckets, gConfig['layer_size'], gConfig['num_layers'], gConfig['max_gradient_norm'], gConfig['batch_size'], gConfig['learning_rate'], gConfig['learning_rate_decay_factor'], forward_only=forward_only,
  vector_src=gConfig['vector_src'])

  if 'pretrained_model' in gConfig:
      model.saver.restore(session,gConfig['pretrained_model'])
      return model

  ckpt = tf.train.get_checkpoint_state(gConfig['model_directory'])
  if ckpt and ckpt.model_checkpoint_path:
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      model.saver.restore(session, ckpt.model_checkpoint_path)
      # model.saver.restore(session, "model/test/seq2seq.ckpt-171500")
  else:
      print("Created model with fresh parameters.")
      session.run(tf.global_variables_initializer())
  return model


def train():
  # prepare dataset
  print("Preparing data in %s" % gConfig['working_directory'])
  _, sent_dec_train = data_utils.prepare_custom_data(
                          working_directory=gConfig['working_directory'],
                          train_dec=gConfig['train_sent_dec'],
                          dec_vocabulary_size=gConfig['dec_vocab_size'])
  # setup config to use BFC allocator
  config = tf.ConfigProto()
  config.gpu_options.allocator_type = 'BFC'

  with tf.Session(config=config) as sess:
    # Create model.
    print("Creating %d layers of %d units." % (gConfig['num_layers'], gConfig['layer_size']))
    model = create_model(sess, False)

    # Read data into buckets and compute their sizes.
    print ("Reading training data (limit: %d)."
           % gConfig['max_train_data_size'])
    train_set, _ = read_vec_sent(vec_enc=gConfig['train_vec_enc'],
                              source_info_file=gConfig['train_enc_info_file'],
                              sent_dec=sent_dec_train,
                              target_info_file=gConfig['train_dec_info_file'],
                              max_size=gConfig['max_train_data_size'])
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))  # total # of videos

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(gConfig['log_dir'], graph=sess.graph)

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while model.global_step.eval() <= gConfig['max_num_steps']:
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      step_loss_summary = tf.Summary()
      learning_rate_summary = tf.Summary()

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)

      step_loss_value = step_loss_summary.value.add()
      step_loss_value.tag = "step loss"
      step_loss_value.simple_value = step_loss.astype(float)
      learning_rate_value = learning_rate_summary.value.add()
      learning_rate_value.tag = "learning rate"
      learning_rate_value.simple_value = model.learning_rate.eval().astype(float)

      # Write logs at every iteration
      summary_writer.add_summary(step_loss_summary, model.global_step.eval())
      summary_writer.add_summary(learning_rate_summary, model.global_step.eval())

      step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
      loss += step_loss / gConfig['steps_per_checkpoint']
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % gConfig['steps_per_checkpoint'] == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f loss %.4f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, loss, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(gConfig['model_directory'], "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0

        sys.stdout.flush()


def scorer():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)

    # Load vocabularies.
    # enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])

    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

    _, sent_dec_test = data_utils.prepare_custom_data(
                            working_directory=gConfig['working_directory'],
                            train_dec=gConfig['test_sent_dec'],
                            dec_vocabulary_size=gConfig['dec_vocab_size'])

    test_set, test_info = read_vec_sent(vec_enc=gConfig['test_vec_enc'],
                              source_info_file=gConfig['test_enc_info_file'],
                              sent_dec=sent_dec_test,
                              target_info_file=gConfig['test_dec_info_file'],
                              max_size=gConfig['max_train_data_size'])
    model.batch_size = 1  # We decode one sentence at a time.
    output_captions = []  # output sentences
    gt_captions = []  # ground truth sentences
    output_info = []  # seq_id and ins_id
    for bucket_id in xrange(len(test_set)):
        for sample_id in xrange(len(test_set[bucket_id])):
            data_pair = test_set[bucket_id][sample_id]
            data_info = test_info[bucket_id][sample_id]
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                data_pair, bucket_id, train=False)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, True)
            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out sentence corresponding to outputs.
            output_caption = " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
            output_captions.append(output_caption)

            gt_caption = " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in data_pair[1]])
            gt_captions.append(gt_caption)

            output_info.append(data_info)

    with open(gConfig['result_dir'] + 'output.txt', 'w') as f:
        for item in output_captions:
            f.write("%s\n" % item)
    with open(gConfig['result_dir'] + 'groundtruth.txt', 'w') as f:
        for item in gt_captions:
            f.write("%s\n" % item.replace(' _EOS',''))
    with open(gConfig['result_dir'] + 'output_info.txt', 'w') as f:
        for item in output_info:
            f.write("%s, %s\n" % (item[0], item[1]))

    # calculate metrics
    pred_dict = {idx: [{'image_id':idx,'caption':sent}] for idx, sent in enumerate(output_captions)}
    gt_dict = {idx: [{'image_id':idx,'caption':sent}] for idx, sent in enumerate(gt_captions)}
    id_list = range(len(gt_captions))
    scorer = COCOScorer()
    total_score = scorer.score(gt_dict, pred_dict, id_list)


if __name__ == '__main__':
    if len(sys.argv) - 1:
        gConfig = get_config(sys.argv[1])
    else:
        # get configuration from seq2seq.ini
        gConfig = get_config()

    if not tf.gfile.Exists(gConfig['model_directory']):
        tf.gfile.MakeDirs(gConfig['model_directory'])
    if not tf.gfile.Exists(gConfig['log_dir']):
        tf.gfile.MakeDirs(gConfig['log_dir'])
    if not tf.gfile.Exists(gConfig['result_dir']):
        tf.gfile.MakeDirs(gConfig['result_dir'])

    print('\n>> Mode : %s\n' %(gConfig['mode']))

    if gConfig['mode'] == 'train':
        # start training
        train()
    elif gConfig['mode'] == 'eval':
        scorer()
    else:
        # wrong way to execute "serve"
        #   Use : >> python ui/app.py
        #           uses seq2seq_serve.ini as conf file
        print('Serve Usage : >> python ui/app.py')
        print('# uses seq2seq_serve.ini as conf file')
