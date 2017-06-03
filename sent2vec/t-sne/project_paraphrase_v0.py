# t-SNE is not available for this task. We directly calculate the distance btw vectors.

import os
import sys
import csv
import numpy as np
import json
import random
import pickle as pkl
import matplotlib.pyplot as plt
from tsne import tsne
from tqdm import *
import pdb
import math
from sklearn.decomposition import PCA

# sent2vec
vec_A_path = '../vectors/caption_1024units_noatt_50kvocab/SICK_enc_cleaned.pkl'
vec_B_path = '../vectors/caption_1024units_noatt_50kvocab/SICK_dec_cleaned.pkl'
# skipthought
vec_C_path = '../data/skipthought/SICK_enc_cleaned_skipthoughts_vec.npy'
vec_D_path = '../data/skipthought/SICK_dec_cleaned_skipthoughts_vec.npy'
# skipgram
vec_E_path = '../data/skipgram/SICK_enc_cleaned_skipgram.txt'
vec_F_path = '../data/skipgram/SICK_dec_cleaned_skipgram.txt'

with open('../sentences/SICK_enc_cleaned.txt', 'r') as f:
    enc_sent = f.readlines()
with open('../sentences/SICK_dec_cleaned.txt', 'r') as f:
    dec_sent = f.readlines()

def proj_PCA(vec_A, vec_B):

    all_data = np.vstack((vec_A,vec_B))

    X = np.array(all_data)
    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.transform(X)

    # X = list(X)
    A = X[:len(X)/2]
    B = X[len(X)/2:]
    Ax = A[:,0]
    Ay = A[:,1]
    Bx = B[:,0]
    By = B[:,1]
    return Ax, Ay, Bx, By

# sent2vec
with open(vec_A_path, 'r') as f:
    vec_A = pkl.load(f)
    del vec_A[0]
    vec_A = np.array(vec_A)
with open(vec_B_path, 'r') as f:
    vec_B = pkl.load(f)
    del vec_B[0]
    vec_B = np.array(vec_B)
Ax, Ay, Bx, By = proj_PCA(vec_A, vec_B)

# # skipthought
# vec_C = np.load(vec_C_path)
# vec_D = np.load(vec_D_path)
# Cx, Cy, Dx, Dy = proj_PCA(vec_C, vec_D)
#
# # skipgram
# with open(vec_E_path, 'r') as f:
#     data = f.readlines()
#     del data[0]  # remove title
#     short_data = np.zeros(100, dtype=np.float)
#     for i in xrange(len(data)):
#         line = data[i].split()
#         del line[0]
#         tmp = np.array(line).astype(np.float)
#         short_data = np.vstack((short_data, tmp))
# vec_E = np.delete(short_data, 0, 0)
# with open(vec_F_path, 'r') as f:
#     data = f.readlines()
#     del data[0]  # remove title
#     short_data = np.zeros(100, dtype=np.float)
#     for i in xrange(len(data)):
#         line = data[i].split()
#         del line[0]
#         tmp = np.array(line).astype(np.float)
#         short_data = np.vstack((short_data, tmp))
# vec_F = np.delete(short_data, 0, 0)
# Ex, Ey, Fx, Fy = proj_PCA(vec_E, vec_F)


num_pairs = [8,10,40,58,74,1224,127]
word = [str(i) for i in num_pairs]
enc_word = [enc_sent[i] for i in num_pairs]
dec_word = [dec_sent[i] for i in num_pairs]
# pdb.set_trace()
enc_word = [str(i)+'A' for i in range(7)]
dec_word = [str(i)+'B' for i in range(7)]

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(Ax[num_pairs], Ay[num_pairs], s=12**2, c='r', marker='o', linewidth='0')
ax1.scatter(Bx[num_pairs], By[num_pairs], s=12**2, c='r', marker='o', linewidth='0')
ax1.plot([Ax[num_pairs], Bx[num_pairs]], [Ay[num_pairs], By[num_pairs]], c='black')
# for i, txt in enumerate(word):
#     ax1.annotate(txt, (Ax[num_pairs][i],Ay[num_pairs][i]))
for i, txt in enumerate(enc_word):
    ax1.annotate(txt, (Ax[num_pairs][i],Ay[num_pairs][i]),fontsize=16)
for i, txt in enumerate(dec_word):
    ax1.annotate(txt, (Bx[num_pairs][i],By[num_pairs][i]),fontsize=16)

# ax1.scatter(Cx[num_pairs], Cy[num_pairs], s=12**2, c='b', marker='s', linewidth='0', label='sentence C')
# ax1.scatter(Dx[num_pairs], Dy[num_pairs], s=12**2, c='b', marker='s', linewidth='0', label='sentence D')
# ax1.plot([Cx[num_pairs], Dx[num_pairs]], [Cy[num_pairs], Dy[num_pairs]], c='black')
# for i, txt in enumerate(word):
#     ax1.annotate(txt, (Cx[num_pairs][i],Cy[num_pairs][i]))
#
# ax1.scatter(Ex[num_pairs], Ey[num_pairs], s=12**2, c='g', marker='^', linewidth='0', label='sentence E')
# ax1.scatter(Fx[num_pairs], Fy[num_pairs], s=12**2, c='g', marker='^', linewidth='0', label='sentence F')
# ax1.plot([Ex[num_pairs], Fx[num_pairs]], [Ey[num_pairs], Fy[num_pairs]], c='black')
# for i, txt in enumerate(word):
#     ax1.annotate(txt, (Ex[num_pairs][i],Ey[num_pairs][i]))



plt.show()
