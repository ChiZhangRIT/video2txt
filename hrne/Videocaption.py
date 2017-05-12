import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb, pdb
# from IPython.core.debugger import Tracer
# from ptbtokenizer import PTBTokenizer
import time
import json
from collections import defaultdict
from tensorflow.python.ops import rnn, rnn_cell
from keras.preprocessing import sequence
from cocoeval import COCOScorer
import unicodedata
from variables import *

def _orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def orthogonal_initializer(scale=1.0):
    """Orthogonal initializer by Saxe et al.
       This initialization is recommended for initializing the
       hidden weights in a RNN.
    References:
        From Lasagne and Keras.
        Paper: Saxe et al., http://arxiv.org/abs/1312.6120
    Parameters
    ----------
    scale: float, optional
        The scale of the orthogonal values.
    Returns
    ----------
    _initializer: function
        Returns the init function.
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        q = _orthogonal(shape)

        return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer
	
def multiple_gaussians(x, W_mu,W_sigma,b_mu,b_sigma, h_prev,number_gaussians,length_elements,alphas_last_dim):

    mu = tf.matmul(W_mu, h_prev)+b_mu #1x100
    mu = tf.div(tf.abs(mu),tf.abs(mu)+1.0)
    sigma = tf.matmul(W_sigma, h_prev)+b_sigma #1x100
    sigma = tf.div(tf.abs(sigma),tf.abs(sigma)+1.0)
    mu = tf.tile(tf.expand_dims(mu,1),tf.pack([1, length_elements,1]))
    first = tf.div(1.0,tf.mul(sigma +epsilon, tf.sqrt(tf.mul(2.0, pi))))

    second_num = tf.mul(-1.0, tf.pow(tf.sub(x, mu),2))
    second_denom = tf.mul(2.0, tf.pow(sigma,2) )
    first = tf.tile(tf.expand_dims(first,1),tf.pack([1, length_elements,1]))
    second_denom = tf.tile(tf.expand_dims(second_denom,1),tf.pack([1, length_elements,1]))
    second = tf.exp(tf.div(second_num,second_denom))
    e_hat_exp =tf.mul(first, second) #2x50x100

    denomin = tf.reduce_sum(e_hat_exp,1) # b
    denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
    denomin = tf.tile(tf.expand_dims(denomin,1),tf.pack([1, length_elements,1]))
    e_hat_exp = tf.div(e_hat_exp,denomin) #2x50x100
    alphas = tf.tile(tf.expand_dims(e_hat_exp, 3),[1, 1, 1, alphas_last_dim]) # n_gaussians x nxb x h  # normalize to obtain alpha

    alphas = tf.reduce_sum(alphas, 0)
    alphas = tf.div(alphas, tf.cast(number_gaussians,tf.float32))

    return alphas
	
class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, beam_size,dim_embedding, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate
        self.beam_size = beam_size
        self.dim_embedding = dim_embedding
        # with tf.device("/cpu:0"):
        self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embedding], -0.1, 0.1), name='Wemb')

        self.lstm1 = rnn_cell.LSTMCell(self.dim_hidden,use_peepholes = False,initializer=orthogonal_initializer(scale=0.2), state_is_tuple=True)
        self.lstm1_dropout = rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.drop_out_rate)

        self.lstm2 = rnn_cell.LSTMCell(self.dim_hidden,use_peepholes = False, initializer=orthogonal_initializer(scale=0.2), state_is_tuple=True)
        self.lstm2_dropout = rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.drop_out_rate)

        self.lstm33 = rnn_cell.LSTMCell(dim_hidden_gaussian,use_peepholes = False, initializer=orthogonal_initializer(scale=0.2),state_is_tuple=False)
        self.lstm3 = rnn_cell.MultiRNNCell([self.lstm33]*2, state_is_tuple=False)
        self.lstm3_dropout = rnn_cell.DropoutWrapper(self.lstm3,output_keep_prob=1 - self.drop_out_rate)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_embedding], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_embedding]), name='encode_image_b')

        #First attention between the visual input and the LSTM filter
        self.embed_att_w_1 = tf.Variable(tf.random_uniform([dim_embedding, 1], -0.1,0.1), name='embed_att_w_1')
        self.embed_att_Wa_1 = tf.Variable(tf.random_uniform([dim_embedding, dim_embedding], -0.1,0.1), name='embed_att_Wa_1')
        self.embed_att_Ua_1 = tf.Variable(tf.random_uniform([dim_hidden_hrne_layer, dim_embedding],-0.1,0.1), name='embed_att_Ua_1')
        self.embed_att_ba_1 = tf.Variable( tf.zeros([dim_embedding]), name='embed_att_ba_1')
        ########################################################################################
        #Second attention between the output of the filter and the second LSTM layer
        self.embed_att_w_2 = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1,0.1), name='embed_att_w_2')
        self.embed_att_Wa_2 = tf.Variable(tf.random_uniform([dim_hidden_hrne_layer, dim_hidden_hrne_layer], -0.1,0.1), name='embed_att_Wa_2')
        self.embed_att_Ua_2 = tf.Variable(tf.random_uniform([dim_hidden_hrne_layer, dim_hidden_hrne_layer],-0.1,0.1), name='embed_att_Ua_2')
        self.embed_att_ba_2 = tf.Variable( tf.zeros([dim_hidden_hrne_layer]), name='embed_att_ba_2')
        ########################################################################################

        #Third attention between the output of our HRNE and the description decoder
        self.W_mu_3 = tf.Variable(tf.div(tf.random_uniform([no_gaussians_third_layer, dim_hidden_gaussian],-0.1,0.1), tf.sqrt(tf.add(1.0, dim_hidden))), name='W_mu_3') #1x512
        self.W_sigma_3 = tf.Variable(tf.div(tf.random_uniform([no_gaussians_third_layer, dim_hidden_gaussian],-0.1,0.1), tf.sqrt(tf.add(1.0, dim_hidden))), name='W_sigma_3')

        self.b_mu_3 = tf.Variable( tf.mul(0.5, tf.cast(tf.constant(1,no_gaussians_third_layer), tf.float32)), name='b_mu_3')
        self.b_sigma_3 = tf.Variable( tf.mul(1.0, tf.cast(tf.constant(1,no_gaussians_third_layer), tf.float32)), name='b_sigma_3')
        ########################################################################################
        self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.embed_nn_Wp = tf.Variable(tf.random_uniform([dim_hidden_gaussian + dim_hidden+dim_embedding, dim_embedding], -0.1,0.1), name='embed_nn_Wp')
        self.embed_nn_bp = tf.Variable(tf.zeros([dim_embedding]), name='embed_nn_bp')

        self.X = tf.expand_dims(tf.expand_dims(tf.linspace(0.0,1.0,length_chain_third_LSTM),1),0)#1xn_
        self.X= tf.cast(tf.tile(self.X, tf.pack([no_gaussians_third_layer, 1 , batch_size])),tf.float32)

    def build_model(self):

        video = tf.placeholder(tf.float32, [self.batch_size, n_total_frames, self.dim_image]) # b x n x d = 100x160x2048
        video_mask = tf.placeholder(tf.float32, [self.batch_size, n_total_frames]) # b x n =100x160
        video_mask_splits = tf.split(1,n_total_frames/stride,video_mask)
        caption = tf.placeholder(tf.int32, [self.batch_size, length_chain_third_LSTM]) # b x 35
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, length_chain_third_LSTM]) # b x 35
        current_embed = tf.zeros([self.batch_size, dim_embedding]) # b x h

        video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x n) x d
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (b x n) x h
        image_emb = tf.reshape(image_emb, [self.batch_size, n_total_frames, dim_embedding]) # b x n x h
        image_emb = tf.transpose(image_emb, [1,0,2]) # n x b x h
        image_emb_splits = tf.split(0,n_total_frames/stride,image_emb)

        state_1 = self.lstm1.zero_state(self.batch_size, tf.float32)
        h_prev_1 = tf.zeros([self.batch_size, self.dim_hidden]) + tf.constant(0.02) # b x h = 100x1024

        state_2 = self.lstm2.zero_state(self.batch_size, tf.float32)
        h_prev_2 = tf.zeros([self.batch_size, self.dim_hidden]) + tf.constant(0.02) # b x h

        state_3 = self.lstm3.zero_state(self.batch_size, tf.float32)
        h_prev_3 = tf.zeros([self.batch_size, dim_hidden_gaussian]) + tf.constant(0.02) # b x h

        loss_caption = 0.0

        brcst_w_1 = tf.tile(tf.expand_dims(self.embed_att_w_1, 0), [length_chain_first_LSTM,1,1]) # n x h x 1
        brcst_w_2 = tf.tile(tf.expand_dims(self.embed_att_w_2, 0), [length_chain_second_LSTM,1,1]) # n x h x 1
        output_first_LSTM = []

        with tf.variable_scope("LSTM2"):
            output2, state_2 = self.lstm2_dropout(tf.zeros([self.batch_size, 2*self.dim_hidden]),state_2)

        with tf.variable_scope("LSTM3"):
            output3, state_3 = self.lstm3_dropout(tf.zeros([self.batch_size, self.dim_hidden+dim_embedding]),state_3)

        for j in range(length_chain_second_LSTM):
            image_part = tf.batch_matmul(image_emb_splits[j], tf.tile(tf.expand_dims(self.embed_att_Wa_1, 0), [length_chain_first_LSTM,1,1])) + self.embed_att_ba_1 # n x b x embedding_size = 8x100x512
            input_visual_embedding_list = tf.split(0,length_chain_first_LSTM, image_emb_splits[j])
            for i in range(length_chain_first_LSTM):
                #Calculate attention between the visual input and the LSTM filter
                e = tf.tanh(tf.matmul(h_prev_1, self.embed_att_Ua_1) + image_part) # n x b x h
                e = tf.batch_matmul(e, brcst_w_1)    # unnormalized relevance score
                e = tf.reduce_sum(e,2) # n x b = 8x100
                e_hat_exp = tf.mul(tf.transpose(video_mask_splits[j]), tf.exp(e)) # n x b
                denomin = tf.reduce_sum(e_hat_exp,0) # b
                denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
                alphas_1 = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,dim_embedding]) # n x b x h  # normalize to obtain alpha

                attention_list_1 = tf.mul(alphas_1, image_emb_splits[j]) # n x b x h = 8x100x512
                atten_1 = tf.reduce_sum(attention_list_1,0) # b x h  =100x512     #  soft-attention weighted sum
                if i>0: tf.get_variable_scope().reuse_variables()
                input_feats_embedding = tf.reshape(input_visual_embedding_list[i], [self.batch_size, -1])
                with tf.variable_scope("LSTM1"):
                    output1, state_1 = self.lstm1_dropout(tf.concat(1,[atten_1, input_feats_embedding]),state_1)
                h_prev_1 = output1
            output_first_LSTM.append(output1)

        input_second_LSTM = tf.pack(output_first_LSTM)
        lsmt1_part = tf.batch_matmul(input_second_LSTM, tf.tile(tf.expand_dims(self.embed_att_Wa_2,0),[length_chain_second_LSTM,1,1])) + self.embed_att_ba_2
        for j in range(length_chain_second_LSTM):
            e_2 = tf.tanh(tf.matmul(h_prev_2, self.embed_att_Ua_2) + lsmt1_part) # b x h
            e_2 = tf.batch_matmul(e_2, brcst_w_2)    # unnormalized relevance score
            e_2 = tf.reduce_sum(e_2,2)
            e_hat_exp_2 =  tf.exp(e_2) # n x b
            denomin_2 = tf.reduce_sum(e_hat_exp_2,0) # b
            denomin_2 = denomin_2 + tf.to_float(tf.equal(denomin_2, 0))   # regularize denominator
            alphas_2 = tf.tile(tf.expand_dims(tf.div(e_hat_exp_2,denomin_2),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
            
            attention_list_2 = tf.mul(alphas_2, input_second_LSTM) # n x b x h
            atten_2 = tf.reduce_sum(attention_list_2,0) # b x h       #  soft-attention weighted sum
            if j>0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM2"):
                output2, state_2 = self.lstm2_dropout(tf.concat(1,[atten_2, output_first_LSTM[j]]),state_2)
            h_prev_2 = output2 #=100x1024

        input_third_LSTM = output2

        for k in range(length_chain_third_LSTM):
            # #Calculate attention between the output of our HRNE and the description decoder
            h_prev_3 = tf.transpose(h_prev_3)
            alphas_3 = multiple_gaussians(self.X, self.W_mu_3,self.W_sigma_3,self.b_mu_3,self.b_sigma_3,h_prev_3,no_gaussians_third_layer,length_chain_third_LSTM,dim_hidden_hrne_layer)

            attention_list_3 = tf.mul(alphas_3, input_third_LSTM) # n x b x h

            atten_3 = tf.reduce_sum(attention_list_3,0) # b x h       #  soft-attention weighted sum
            if k>0: tf.get_variable_scope().reuse_variables()
            #
            with tf.variable_scope("LSTM3"):
                output3, state_3 = self.lstm3_dropout(tf.concat(1,[atten_3, current_embed]),state_3)

            h_prev_3 = output3 # b x h
            output_final = tf.tanh(tf.nn.xw_plus_b(tf.concat(1,[output3,atten_3,current_embed]), self.embed_nn_Wp, self.embed_nn_bp)) # b x h

            labels = tf.expand_dims(caption[:,k], 1) # b x 1
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
            concated = tf.concat(1, [indices, labels]) # b x 2
            onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w
            current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,k])

            logit_words = tf.nn.xw_plus_b(output_final, tf.transpose(self.Wemb), self.embed_word_b) # b x w
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels) # b x 1
            cross_entropy = cross_entropy * caption_mask[:,k] # b x 1
            loss_caption += tf.reduce_sum(cross_entropy) # 1

        loss_caption = loss_caption / tf.reduce_sum(caption_mask)
        tf.summary.scalar("loss", loss_caption)
        summary_op =  tf.summary.merge_all()


        return loss_caption, video, video_mask, caption, caption_mask, summary_op


    def build_generator(self):
        generated_words = []
        generated_words_value = []
        gather_path_indices = []

        video = tf.placeholder(tf.float32, [self.batch_size, n_total_frames, self.dim_image]) # b x n x d = 100x160x2048
        video_mask = tf.placeholder(tf.float32, [self.batch_size, n_total_frames]) # b x n =100x160
        video_mask_splits = tf.split(1,n_total_frames/stride,video_mask)
        caption = tf.placeholder(tf.int32, [self.batch_size, length_chain_third_LSTM]) # b x 35
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, length_chain_third_LSTM]) # b x 35
        current_embed = tf.zeros([self.batch_size, dim_embedding]) # b x h


        video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x n) x d
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (b x n) x h
        image_emb = tf.reshape(image_emb, [self.batch_size, n_total_frames, dim_embedding]) # b x n x h
        image_emb = tf.transpose(image_emb, [1,0,2]) # n x b x h
        image_emb_splits = tf.split(0,n_total_frames/stride,image_emb)

        state_1 = self.lstm1.zero_state(self.batch_size, tf.float32)
        h_prev_1 = tf.zeros([self.batch_size, self.dim_hidden]) + tf.constant(0.02) # b x h = 100x1024

        state_2 = self.lstm2.zero_state(self.batch_size, tf.float32)
        h_prev_2 = tf.zeros([self.batch_size, self.dim_hidden]) + tf.constant(0.02) # b x h

        state_3 = self.lstm3.zero_state(self.batch_size, tf.float32)
        h_prev_3 = tf.zeros([self.batch_size, dim_hidden_gaussian]) + tf.constant(0.02) # b x h

        loss_caption = 0.0

        brcst_w_1 = tf.tile(tf.expand_dims(self.embed_att_w_1, 0), [length_chain_first_LSTM,1,1]) # n x h x 1
        brcst_w_2 = tf.tile(tf.expand_dims(self.embed_att_w_2, 0), [length_chain_second_LSTM,1,1]) # n x h x 1
        output_first_LSTM = []

        with tf.variable_scope("LSTM2"):
            output2, state_2 = self.lstm2_dropout(tf.zeros([self.batch_size, 2*self.dim_hidden]),state_2)

        with tf.variable_scope("LSTM3"):
            output3, state_3 = self.lstm3_dropout(tf.zeros([self.batch_size, self.dim_hidden+dim_embedding]),state_3)

        for j in range(length_chain_second_LSTM):
            image_part = tf.batch_matmul(image_emb_splits[j], tf.tile(tf.expand_dims(self.embed_att_Wa_1, 0), [length_chain_first_LSTM,1,1])) + self.embed_att_ba_1 # n x b x embedding_size = 8x100x512
            input_visual_embedding_list = tf.split(0,length_chain_first_LSTM, image_emb_splits[j])
            for i in range(length_chain_first_LSTM):
                #Calculate attention between the visual input and the LSTM filter
                e = tf.tanh(tf.matmul(h_prev_1, self.embed_att_Ua_1) + image_part) # n x b x h
                e = tf.batch_matmul(e, brcst_w_1)    # unnormalized relevance score
                e = tf.reduce_sum(e,2) # n x b = 8x100
                e_hat_exp = tf.mul(tf.transpose(video_mask_splits[j]), tf.exp(e)) # n x b
                denomin = tf.reduce_sum(e_hat_exp,0) # b
                denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
                alphas_1 = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,dim_embedding]) # n x b x h  # normalize to obtain alpha

                attention_list_1 = tf.mul(alphas_1, image_emb_splits[j]) # n x b x h = 8x100x512
                atten_1 = tf.reduce_sum(attention_list_1,0) # b x h  =100x512     #  soft-attention weighted sum
                if i>0: tf.get_variable_scope().reuse_variables()
                input_feats_embedding = tf.reshape(input_visual_embedding_list[i], [self.batch_size, -1])
                with tf.variable_scope("LSTM1"):
                    output1, state_1 = self.lstm1_dropout(tf.concat(1,[atten_1, input_feats_embedding]),state_1)
                h_prev_1 = output1
            output_first_LSTM.append(output1)

        input_second_LSTM = tf.pack(output_first_LSTM)
        lsmt1_part = tf.batch_matmul(input_second_LSTM, tf.tile(tf.expand_dims(self.embed_att_Wa_2,0),[length_chain_second_LSTM,1,1])) + self.embed_att_ba_2
        for j in range(length_chain_second_LSTM):
            # h_prev_2 = tf.transpose(h_prev_2)
            e_2 = tf.tanh(tf.matmul(h_prev_2, self.embed_att_Ua_2) + lsmt1_part) # b x h
            e_2 = tf.batch_matmul(e_2, brcst_w_2)    # unnormalized relevance score
            e_2 = tf.reduce_sum(e_2,2)
            e_hat_exp_2 =  tf.exp(e_2) # n x b
            denomin_2 = tf.reduce_sum(e_hat_exp_2,0) # b
            denomin_2 = denomin_2 + tf.to_float(tf.equal(denomin_2, 0))   # regularize denominator
            alphas_2 = tf.tile(tf.expand_dims(tf.div(e_hat_exp_2,denomin_2),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
            
            attention_list_2 = tf.mul(alphas_2, input_second_LSTM) # n x b x h
            atten_2 = tf.reduce_sum(attention_list_2,0) # b x h       #  soft-attention weighted sum
            if j>0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM2"):
                output2, state_2 = self.lstm2_dropout(tf.concat(1,[atten_2, output_first_LSTM[j]]),state_2)
            h_prev_2 = output2 #=100x1024

        input_third_LSTM = output2

        current_embed = tf.zeros([self.batch_size, dim_embedding]) # b x h
        temp_h_prev_3 = []
        temp_state_3 = []

        pack_gaussians = (self.X, self.W_mu_3, self.W_sigma_3, self.b_mu_3, self.b_sigma_3, self.Wemb, self.lstm3_dropout, self.embed_nn_Wp, self.embed_nn_bp, self.embed_word_b)
        current_embed = tf.zeros([self.batch_size, dim_embedding]) # b x h
        prev_probs = tf.ones([self.batch_size,beam_size])
        for k in xrange(length_chain_third_LSTM):
            if k==0:
                top_value, top_index, h_prev_3, state_3, stack_current_embed = beam_search(prev_probs, k, input_third_LSTM, current_embed, h_prev_3, self.beam_size, state_3, pack_gaussians)

                generated_words.append(top_index)
                generated_words_value.append(top_value)


                prev_h_prev_3 = [h_prev_3] * beam_size
                prev_state_3 = [state_3] * beam_size
                prev_current_embed = stack_current_embed
                prev_probs = top_value

            else:
                #From previous top 3 words pick the next top 9 words
                #
                stack_temp_stack_current_embed = []
                stack_temp_h_prev_3 = []
                temp_h_prev_3 = [None] * self.beam_size
                temp_state_3 = [None] * self.beam_size
                #

                for kk in xrange(self.beam_size):
                    top_value, top_index, temp_h_prev_3[kk], temp_state_3[kk], temp_stack_current_embed = beam_search(prev_probs, k, input_third_LSTM, prev_current_embed[kk], prev_h_prev_3[kk], self.beam_size, prev_state_3[kk], pack_gaussians)
                    if kk ==0 :
                        temp_top_value = top_value
                        temp_top_index = top_index
                        stack_temp_h_prev_3 = [temp_h_prev_3[kk]]*self.beam_size
                        stack_temp_state_3 = [temp_state_3[kk]]*self.beam_size
                        #
                    else:
                        temp_top_value = tf.concat(1, [temp_top_value, top_value])
                        temp_top_index = tf.concat(1, [temp_top_index, top_index])
                        stack_temp_h_prev_3 = stack_temp_h_prev_3 + [temp_h_prev_3[kk]]*self.beam_size
                        stack_temp_state_3 = stack_temp_state_3 +  [temp_state_3[kk]]*self.beam_size
                    stack_temp_stack_current_embed = stack_temp_stack_current_embed + temp_stack_current_embed
                #
                # Pick top 3 from 9 words base on prob value
                index_to_pick = []
                val_to_pick = []
                for ii in xrange(self.batch_size):
                    v, t = tf.nn.top_k(temp_top_value[ii,:], k = beam_size, sorted=True)
                    index_to_pick.append(t)
                    val_to_pick.append(v)
                index_to_pick = tf.pack(index_to_pick)
                prev_probs = tf.pack(val_to_pick)
                #

                gather_path_indices.append(tf.div(index_to_pick,beam_size))
                #Pick top 3 words along with current_embed
                stack_temp_stack_current_embed = tf.unpack(tf.transpose(tf.pack(stack_temp_stack_current_embed),[1,0,2]))
                stack_temp_h_prev_3 = tf.unpack(tf.transpose(tf.pack(stack_temp_h_prev_3),[1,0,2]))

                stack_temp_state_3 = tf.unpack(tf.transpose(tf.pack(stack_temp_state_3),[1,0,2]))

                temp_prev_current_embed = []
                temp_prev_h_prev_3 = []
                temp_state_3 = []
                out_top_index = []
                out_top_value = []
                for ii in xrange(self.batch_size):
                    temp_prev_current_embed.append(tf.gather(stack_temp_stack_current_embed[ii], index_to_pick[ii]))
                    temp_prev_h_prev_3.append(tf.gather(stack_temp_h_prev_3[ii], index_to_pick[ii]))
                    temp_state_3.append(tf.gather(stack_temp_state_3[ii], index_to_pick[ii]))
                #Prepare for next word
                prev_current_embed = tf.transpose(tf.pack(temp_prev_current_embed),[1,0,2])
                prev_h_prev_3 = tf.transpose(tf.pack(temp_prev_h_prev_3),[1,0,2])
                prev_state_3 = tf.transpose(tf.pack(temp_state_3),[1,0,2])
                temp_top_index = tf.unpack(temp_top_index)
                temp_top_value = tf.unpack(temp_top_value)
                for ii in xrange(self.batch_size):
                    out_top_index.append(tf.gather(temp_top_index[ii], index_to_pick[ii]))
                    out_top_value.append(tf.gather(temp_top_value[ii], index_to_pick[ii]))

                out_top_index = tf.pack(out_top_index)
                out_top_value = tf.pack(out_top_value)
                
                generated_words.append(out_top_index)
                generated_words_value.append(out_top_value)
        generated_words = tf.transpose(tf.pack(generated_words),[1,0,2])
        generated_words_value = tf.transpose(tf.pack(generated_words_value),[1,0,2])
        return video, video_mask, generated_words, generated_words_value, gather_path_indices