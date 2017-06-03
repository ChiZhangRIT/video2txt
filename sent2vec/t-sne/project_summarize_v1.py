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

# directory of saving output images. If show image, set to None.
dir_save_im = None  # or 'tsne_skipthought.png'
# path to tacos descriptions - short can be "short" or "detailed" depending on what you want to plot.
# short_desc_path = '../data/skipthought/tacos_singlesentence_test_vec_encoder.np'
# short_desc_path = '../data/skipgram/tacos_singlesentence_test.txt.vec'
short_desc_path = '../vectors/caption_300units_noatt_50kvocab/tacos_singlesentence_test.pkl'
short_info_path = '../sentences/tacos_singlesentence_info_test.json'

def build_tacos(short_src,
                short_info_path,
                num_sequence=42,  # max number of samples is 42 in test set
                pt_limit=20):     # number of samples oer class

    if short_src[-2:] == 'np':
        short_data = np.load(short_src)  # numpy array
    elif short_src[-3:] == 'pkl':
        with open(short_src, 'r') as f:
            short_data = np.asarray(pkl.load(f))  # list tp numpy array
    elif short_src[-3:] == 'vec':  # skipgram txt file
        with open(short_src, 'r') as f:
            data = f.readlines()
            del data[0]  # remove title
            short_data = np.zeros(100, dtype=np.float)
            for i in xrange(len(data)):
                line = data[i].split()
                del line[0]
                tmp = np.array(line).astype(np.float)
                short_data = np.vstack((short_data, tmp))
        short_data = np.delete(short_data, 0, 0)
    else:
        raise ValueError("Wrong vector file.")

    with open(short_info_path, 'r') as f:
        short_info = json.load(f)

    # unique_seq_id = list(set(short_info['sequence_id_list']))
    # sent2vec
    unique_seq_id = [u's22-d43', u's28-d23', u's29-d29', u's33-d54', u's22-d25',  u's34-d40', u's29-d49', u's29-d71', u's28-d46',  u's22-d34',  u's22-d48', u's34-d73', u's34-d34']
    # skipthought
    # unique_seq_id =  [u's34-d73', u's22-d25', u's34-d34', u's28-d27', u's28-d25', u's28-d23', u's22-d43', u's22-d35', u's29-d49', u's33-d54', u's22-d55', u's22-d31', u's29-d29', u's34-d63', u's34-d40']
    num_sequence = len(unique_seq_id)

    mask = []
    mat = []
    seq_ids = []
    for i in trange(num_sequence, desc='Extracting sequences'):
        seq_id = unique_seq_id[i]
        seq_inds = [j for j, item in enumerate(short_info['sequence_id_list'])
                    if item == seq_id]

        # limit the number of points in each category <= pt_limit
        if len(seq_inds) > pt_limit:
            seq_inds = random.sample(set(seq_inds), pt_limit)

        short = list(short_data[seq_inds, :])

        mat += short
        mask += [i] * len(short)
        seq_ids += [seq_id] * len(short)
    return (mat, mask, seq_ids)

def tsne_viz(
        mat=None,
        mask=None,
        seq_ids = None,
        indices=None,
        output_filename=None,
        figheight=40,
        figwidth=50,
        display_progress=False):

    num_sequence = len(set(mask))
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
    colors = [colormap(np.linspace(0, 0.9, num_sequence)[i]) for i in mask]

    # _mask = [None] + mask
    # marker_tmp = [1 if _mask[i] != _mask[i-1] else 0 for i in xrange(1, len(_mask))]
    # markers = ['^' if marker_tmp[i] == 1 else 'o' for i in xrange(len(marker_tmp))]
    markers = ['o'] * len(mat)
    # marker_sizes = [20*4**2 if marker_tmp[i] == 1 else 20*4**1 for i in xrange(len(marker_tmp))]
    marker_sizes = [12*4**1] * len(mat)

    temp = sys.stdout
    if not display_progress:
        # Redirect stdout so that tsne doesn't fill the screen with its iteration info:
        f = open(os.devnull, 'w')
        sys.stdout = f

    print "Running t-SNE on the dataset..."

    tsnemat = tsne(mat)
    sys.stdout = temp
    # Plot coordinates:
    if not indices:
        indices = range(len(mat))
    xvals = tsnemat[indices, 0]  # projected x-values
    yvals = tsnemat[indices, 1]  # projected y-values
    # Plotting:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(100)
    fig.set_figwidth(500)
    for _s, _marker, _c, _x, _y in zip(marker_sizes, markers, colors, xvals, yvals):
        ax.scatter(_x, _y, s=_s, c=_c, marker=_marker, linewidth='0')

    # for word, x, y, color in zip(seq_ids, xvals, yvals, colors):
    #     ax.annotate(word, (x, y), fontsize=8, color=color)

    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
        print "Image saved."
    else:
        plt.show()


if __name__ == "__main__":

    wv = build_tacos(short_desc_path,
                     short_info_path)

    tsne_viz(mat=np.array(wv[0]),
             mask=wv[1],
             seq_ids= wv[2],
             output_filename=dir_save_im,
             display_progress=True)
