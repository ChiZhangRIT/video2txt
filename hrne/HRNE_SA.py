#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import pdb
import time
import json
import pickle as pkl
from tqdm import *
from collections import defaultdict
from tensorflow.python.ops import rnn, rnn_cell
from keras.preprocessing import sequence
from cocoeval import COCOScorer
from variables import *
from data_utils import *
import unicodedata
# import Videocaption
from Videocaption import Video_Caption_Generator

try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser # In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.

# use a number of buckets and pad to the closest one for efficiency. Once buckets are changed, the training process need to put samples into buckets again. Run helper/num_sent_tacos.py to plot histogram of lengths of sentences, if needed.
buckets = [20, 40]

gConfig = {}
def get_config(config_file='arguments.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats, strings and booleans
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    _conf_booleans = [ (name, parser.getboolean('booleans', name))
                        for name in parser.options('booleans') ]
    return dict(_conf_ints + _conf_floats + _conf_strings + _conf_booleans)


def beam_search(prev_probs, k, input_third_LSTM, current_embed, h_prev_3, beam_size, *args):
    h_prev_3 = tf.transpose(h_prev_3)
    stack_current_embed = []
    #Unpack state_3 to tuple    #
    state_3 = args[0]
    X, W_mu_3, W_sigma_3, b_mu_3, b_sigma_3, Wemb, lstm3_dropout, embed_nn_Wp, embed_nn_bp, embed_word_b = args[1]

    alphas_3 = multiple_gaussians(X, W_mu_3, W_sigma_3, b_mu_3, b_sigma_3,h_prev_3,no_gaussians_third_layer,length_chain_third_LSTM,dim_hidden_hrne_layer)
    attention_list_3 = tf.mul(alphas_3, input_third_LSTM) # n x b x h
    atten_3 = tf.reduce_sum(attention_list_3,0) # b x h       #  soft-attention weighted sum
    if k>0: tf.get_variable_scope().reuse_variables()


    with tf.variable_scope("LSTM3"):
        output3, state_3 = lstm3_dropout(tf.concat(1,[atten_3, current_embed]),state_3)

    h_prev_3 = output3 # b x h
    output_final = tf.tanh(tf.nn.xw_plus_b(tf.concat(1,[output3,atten_3,current_embed]), embed_nn_Wp, embed_nn_bp)) # b x h

    logit_words = tf.nn.xw_plus_b(output_final, tf.transpose(Wemb), embed_word_b) # b x w

    top_value, top_index = tf.nn.top_k(tf.nn.softmax(logit_words), k=beam_size, sorted = True)

    top_value = tf.mul(prev_probs, top_value)
    #
    for i in xrange(beam_size):
        current_embed = tf.nn.embedding_lookup(Wemb, top_index[:,i])
        stack_current_embed.append(current_embed)

    return top_value, top_index, h_prev_3, state_3, stack_current_embed


def preProBuildWordVocab(sentence_iterator, word_count_threshold=5): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

def testing_one(sess, video_feat_path, ixtoword, video_tf, video_mask_tf, caption_tf, counter, lstm2_tf):
    pred_sent = []
    gt_sent = []
    IDs = []
    namelist = []
    #print video_feat_path
    test_data_batch = h5py.File(video_feat_path)
    gt_captions = json.load(open('msvd_cap_vidid.json'))

    video_feat = np.zeros((batch_size, n_total_frames, dim_image))
    video_mask = np.zeros((batch_size, n_total_frames))

    for ind in xrange(batch_size):
        video_feat[ind,:,:] = test_data_batch['data'][:,0,:][idx_frames_to_pick]
        idx = np.where(test_data_batch['label'][:] != -1)[0]
        if(len(idx) == 0):
            continue
        video_mask[ind,:idx[-1]+1] = 1.
    #
    generated_word_index, lstm2_part = sess.run([caption_tf, lstm2_tf], feed_dict={video_tf:video_feat, video_mask_tf:video_mask})

    cap_key = test_data_batch['fname'][0]
    generated_words = ixtoword[generated_word_index[0]]
    punctuation = np.argmax(np.array(generated_words) == '.')+1
    generated_words = generated_words[:punctuation]

    generated_sentence = ' '.join(generated_words)
    pred_sent.append([{'image_id':str(counter),'caption':generated_sentence}])
    namelist.append(cap_key)
    for i,s in enumerate(gt_captions[cap_key]):
        s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
        gt_sent.append([{'image_id':str(counter),'cap_id':i,'caption':s}])
        IDs.append(str(counter))
    counter += 1
    #
    return pred_sent, gt_sent, IDs, counter, namelist

def testing_all(sess, test_data, ixtoword, video_tf, video_mask_tf, caption_tf, lstm2_tf):
    pred_sent = []
    gt_sent = []
    IDs_list = []
    flist = []
    counter = 0
    gt_dict = defaultdict(list)
    pred_dict = {}
    for _, video_feat_path in enumerate(test_data):
        [b,c,d, counter, fns] = testing_one(sess, video_feat_path, ixtoword, video_tf, video_mask_tf, caption_tf, counter, lstm2_tf)
        pred_sent += b
        gt_sent += c
        IDs_list += d
        flist += fns

    for k,v in zip(IDs_list,gt_sent):
        gt_dict[k].append(v[0])

    new_flist = []
    new_IDs_list = []
    for k,v in zip(range(len(pred_sent)),pred_sent):
        if flist[k] not in new_flist:
            new_flist.append(flist[k])
            new_IDs_list.append(str(k))
            pred_dict[str(k)] = v

    return pred_sent, gt_sent, new_IDs_list, gt_dict, pred_dict

def findpath_indices(start, index, value, connections, iter = 0, path=[]):
    path = path + [value[iter][start]]
    path = path + [index[iter][start]]
    if (index[iter][start] == 0) or (iter == 34):
        return path
    cc = connections[iter]
    if start not in cc:
        return path
    paths = []
    for i,ccc in enumerate(cc):
        if ccc == start:
            child_paths = findpath_indices(i,index, value, connections,iter+1, path)
            paths.append(child_paths)
    return paths

def flatten_list(S,test=[]):
    if (type(S) is list) and (type(S[0]) is not list):
        if S not in test:
            test.append(S)
        return S
    res = []
    for l in S:
        res.append(flatten_list(l,test))
    return res,test

def get_idx_for_one(idx,val,path_idx, beam_size):
    p = []
    acc = []
    for i in xrange(beam_size):
        _,flat = flatten_list(findpath_indices(i,idx, val, path_idx),acc)
        if type(flat) is list:
            p = p + flat

    p = map(list,set(map(tuple,p)))
    # Remove element with len = 1
    # p = [i for i in p if len(i)>2]
    p.sort(key = lambda x :sum(x[0::2]), reverse = True)

    return p

def get_top_sentences(gwi, gwv, gpi, beam_size, batch_size):
    output = []
    for q in xrange(batch_size):
        path =  get_idx_for_one(gwi[q], gwv[q], gpi[:,q,:], beam_size)
        output.append(path)
    return output

def testing_one_batch(sess, video_feat_path, ixtoword, video_tf, video_mask_tf, caption_tf, counter, value_tf, gather_path_indices_tf):
    pred_sent = []
    gt_sent = []
    IDs = []
    namelist = []
    test_data_batch = h5py.File(video_feat_path)
    gt_captions = json.load(open('msvd_cap_vidid.json'))


    video_feat = np.zeros((batch_size, n_total_frames, dim_image))
    video_mask = np.zeros((batch_size, n_total_frames))

    for ind in xrange(batch_size):
        if test_data_batch['fname'].shape[0]!=batch_size:
            if ind>=test_data_batch['fname'].shape[0]:
                to_pick_ind = 0

            else:
                to_pick_ind = ind
            video_feat[ind,:n_total_frames,:] = test_data_batch['data'][idx_frames_to_pick,to_pick_ind,:]
            idx = np.where(test_data_batch['label'][:,to_pick_ind] != -1)[0]
        else:
            video_feat[ind,:n_total_frames,:] = test_data_batch['data'][idx_frames_to_pick,ind,:]
            idx = np.where(test_data_batch['label'][:,ind] != -1)[0]
        # video_feat[ind,:,:] = test_data_batch['data'][:n_frame_step,:]
        if(len(idx) == 0):
            continue
        video_mask[ind,:idx[-1]+1] = 1.
    #
    generated_word_index, generated_word_value, gather_path_indices = sess.run([caption_tf, value_tf, gather_path_indices_tf], feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
    gwi = np.array(generated_word_index)
    gwv = np.array(generated_word_value)
    gpi = np.array(gather_path_indices)


    generated_word_index = get_top_sentences(gwi, gwv, gpi, beam_size, batch_size)



    for ind in xrange(test_data_batch['fname'].shape[0]):
        cap_key = test_data_batch['fname'][ind]
        # for q in xrange(beam_size):
        generated_words = ixtoword[generated_word_index[ind][0][1::2]]
        # punctuation = np.argmax(np.array(generated_words) == '.')+1
        # generated_words = generated_words[:punctuation]

        generated_sentence = ' '.join(generated_words)
        pred_sent.append([{'image_id':str(counter),'caption':generated_sentence}])
        namelist.append(cap_key)
        for i,s in enumerate(gt_captions[cap_key]):
            s = unicodedata.normalize('NFKD', s).encode('ascii','ignore')
            gt_sent.append([{'image_id':str(counter),'cap_id':i,'caption':s}])
            IDs.append(str(counter))
        counter += 1
    #
    return pred_sent, gt_sent, IDs, counter, namelist

def testing_all_batch(sess, test_data, ixtoword, video_tf, video_mask_tf, caption_tf, value_tf, gather_path_indices_tf):
    pred_sent = []
    gt_sent = []
    IDs_list = []
    flist = []
    counter = 0
    gt_dict = defaultdict(list)
    pred_dict = {}
    for _, video_feat_path in enumerate(test_data):
        [b,c,d, counter, fns] = testing_one_batch(sess, video_feat_path, ixtoword, video_tf, video_mask_tf, caption_tf, counter, value_tf, gather_path_indices_tf)
        pred_sent += b
        gt_sent += c
        IDs_list += d
        flist += fns

    for k,v in zip(IDs_list,gt_sent):
        gt_dict[k].append(v[0])

    new_flist = []
    new_IDs_list = []
    for k,v in zip(range(len(pred_sent)),pred_sent):
        if flist[k] not in new_flist:
            new_flist.append(flist[k])
            new_IDs_list.append(str(k))
            pred_dict[str(k)] = v
    return pred_sent, gt_sent, new_IDs_list, gt_dict, pred_dict

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params

def train():
    # load training data
    train_filename = 'data/tacos_train_' + '_'.join(str(i) for i in buckets) + '.pkl'
    train_inputs, train_seq_ids, train_ins_ids = load_data(train_filename, buckets, gConfig['vec_file'], gConfig['vec_info_file'])
    # load single sentence ground truths. Some entries in train_inputs, train_seq_ids, and train_ins_ids will be cleaned if there is no match in singlesentence gt.
    train_inputs, train_seq_ids, train_ins_ids, train_ss_gt = correspond_gt(train_inputs, train_seq_ids, train_ins_ids, gConfig['singlesentence_file'], gConfig['singlesentence_info_file'], buckets, 'singlesentence')

    print '\n'
    pdb.set_trace()

    # meta_data, train_data, val_data, test_data = get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test)

    wordtoix=pd.Series(np.load('./data0/wordtoix-msvd-PTBtokenizer-nochange.npy').tolist())

    current_feats = np.zeros((batch_size, n_total_frames, dim_image))
    current_video_masks = np.zeros((batch_size, n_total_frames))
    current_video_len = np.zeros(batch_size)
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden_hrne_layer,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate = 0.5,
			beam_size = beam_size,
			dim_embedding = dim_embedding,
            bias_init_vector=None)
    print "Finish graph construction"
    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask, tf_summary = model.build_model()

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    print "Finish session initialization"
    with tf.device("/cpu:0"):
        saver = tf.train.Saver(max_to_keep=100)

    train_op = tf.train.AdamOptimizer(learning_rate,  beta1=0.9, beta2=0.999)
    grad_vars = train_op.compute_gradients(tf_loss)
    capped_grad_vars = [(tf.clip_by_value(grad,-1., 1.), var) for grad, var in grad_vars]
    apply_grad_vars = train_op.apply_gradients(capped_grad_vars)
    print "Finish gradients clipping setup"
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    tStart_total = time.time()
    time_monitor = []
    for epoch in range(n_epochs):
        index = np.arange(len(train_data))
        np.random.shuffle(index)
        train_data = train_data[index]

        tStart_epoch = time.time()
        loss_epoch = np.zeros(len(train_data))
        for current_batch_file_idx in xrange(len(train_data)):
            tStart = time.time()
            current_batch = h5py.File(train_data[current_batch_file_idx])
            for ind in xrange(batch_size):
                current_feats[ind,:,:] = current_batch['data'][:,ind,:][idx_frames_to_pick]
                idx = np.where(current_batch['label'][:,ind] != -1)[0]
                if len(idx) == 0:
                    continue
                current_video_masks[ind,:idx[-1]+1] = 1
            #Close unused variable
            current_captions = current_batch['title']
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.split(' ') if word in wordtoix], current_captions)
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=n_caption_step-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            _, loss_val, summary = sess.run(
                    [apply_grad_vars, tf_loss, tf_summary],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })
            loss_epoch[current_batch_file_idx] = loss_val
            writer.add_summary(summary, epoch * len(train_data) + current_batch_file_idx)
            tStop = time.time()
            print "Epoch:", epoch, " Batch:", current_batch_file_idx, " Loss:", loss_val
        print "Time Cost:", round(tStop - tStart,2), "s"
        time_monitor.append("Time Cost: "+ str(round(tStop - tStart,2))+"s")
        print "Epoch:", epoch, " done. Loss:", np.mean(loss_epoch)
        tStop_epoch = time.time()
        print "Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s"
        time_monitor.append("Epoch Time Cost:" + str(round(tStop_epoch - tStart_epoch,2)) + "s")
        if np.mod(epoch, 1) == 0 or epoch == n_epochs - 1:
            print "Epoch ", epoch, " is done. Saving the model ..."
            with tf.device("/cpu:0"):
                saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

        sys.stdout.flush()

    print "Finally, saving the model ..."
    with tf.device("/cpu:0"):
        saver.save(sess, os.path.join(model_path, 'model'), global_step=n_epochs)
    tStop_total = time.time()
    print "Total Time Cost:", round(tStop_total - tStart_total,2), "s"
    time_monitor.append("Total Time Cost:" + str(round(tStop_total - tStart_total,2)) + "s")
    # with open('./monitor_time/'+TODAY+'FRAME_LENGTH'+str(n_total_frames)+'-'.join(list_experiments)+'.time','w') as f:
        # f.write('\n'.join(time_monitor))

def test(model_path='models/model-900', video_feat_path=video_feat_path):
    meta_data, train_data, val_data, test_data = get_video_data_jukin(video_data_path_train, video_data_path_val, video_data_path_test)
    # test_data = val_data   # to evaluate on testing data or validation data
    # ixtoword = pd.Series(np.load('./data0/ixtoword.npy').tolist())
    # ixtoword=pd.Series(np.load('./data0/ixtoword-resnet-fixed-nochange.npy').tolist())
    ixtoword=pd.Series(np.load('./data0/ixtoword-msvd-PTBtokenizer-nochange.npy').tolist())
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden_hrne_layer,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate = 0,
			beam_size = beam_size,
			dim_embedding = dim_embedding,
            bias_init_vector=None)

    # video_tf, video_mask_tf, caption_tf, lstm3_variables_tf = model.build_generator()
    video_tf, video_mask_tf, caption_tf, value_tf, gather_path_indices_tf = model.build_generator()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    with tf.device("/cpu:0"):
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

    tStart = time.time()
    [pred_sent, gt_sent, id_list, gt_dict, pred_dict] = testing_all_batch(sess, test_data, ixtoword,video_tf, video_mask_tf, caption_tf, value_tf, gather_path_indices_tf)
    tStop = time.time()
    timetest = "Total Test Time Cost:" + str(round(tStop - tStart,2)) + "s"+ "\n"
    with open('./monitor_time/'+TODAY+'FRAME_LENGTH'+str(n_total_frames)+'-'.join(list_experiments)+'.timetest','a') as f:
        f.write(timetest)

    np.savez('/home/dp1248/result-'+TODAY+'_'+model_path.split('/')[1]+'_'.join(list_experiments),gt = gt_sent,pred=pred_sent)
    scorer = COCOScorer()
    total_score = scorer.score(gt_dict, pred_dict, id_list)
    return total_score

if __name__ == '__main__':

    if len(sys.argv) - 1:
        gConfig = get_config(sys.argv[1])
    else:
        gConfig = get_config()

    if not os.path.exists(gConfig['model_path']):
        os.makedirs(gConfig['model_path'])
    if not os.path.exists(gConfig['log_path']):
        os.makedirs(gConfig['log_path'])

    print('\n>> Mode : %s\n' %(gConfig['mode']))

    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'test':
        total_score = test(model_path = args.model)
    else:
        print('%s mode not vailable. Mode options: train or test.' %(gConfig['mode']))
