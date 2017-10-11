import numpy as np
import tensorflow as tf

# Paths to dataset
list_experiments = ['HRNE','SoftAttention', 'varlength']
video_data_path_train = '/home/dp1248/h5_125_resnet_corr/train_val_100.txt'
video_data_path_val = '/home/dp1248/h5_125_resnet_corr/val100.txt'
video_data_path_test = '/home/dp1248/h5_125_resnet_corr/test100.txt'
video_feat_path = '/home/PaulChen/h5py_data/cont_augment/'

# gpu_id = 3
dim_image = 2048
dim_hidden_gaussian= 256
n_frame_step = 50 # total number of sampled frames
n_caption_step = 35 # no of units in decoder
batch_size = 100
learning_rate = 0.0001

dim_hidden_hrne_layer = 1024
n_total_frames = 50
stride = 5
MAX_FRAME = 125  # total number of frames in the video
dim_embedding = 512  # linear embedding matrix dimension

length_chain_first_LSTM = stride
length_chain_second_LSTM = n_total_frames/stride
length_chain_third_LSTM = n_caption_step
idx_frames_to_pick = np.linspace(0,MAX_FRAME-1,n_total_frames,dtype=np.int16)

no_gaussians_first_layer = 1
no_gaussians_second_layer = 1
no_gaussians_third_layer = 1
beam_size = 5

pi = tf.cast(3.141592653589793, tf.float32)
epsilon = tf.cast(np.finfo(float).eps, tf.float32)

model_path = 'model/'
logs_path = 'log/'
