import os
import sys
import csv
import numpy as np
import json
from tsne import tsne
import matplotlib.pyplot as plt
import pdb

# directory of saving output images. If show image, set to None.
dir_save_im = 'tsne_skipthought.png'
# path to tacos descriptions
short_desc_path = '../data/skipthought/tacos_short_test_vec_encoder.np'
short_info_path = '../sentences/tacos_short_info_test.json'
sinsent_desc_path = '../data/skipthought/tacos_singlesentence_test_vec_encoder.np'
sinsent_info_path = '../sentences/tacos_singlesentence_info_test.json'


def build_tacos(short_src,
                sinsent_src,
                short_info_path,
                sinsent_info_path,
                sample_idx=range(1, 51)):
    short_data = np.load(short_src)
    sinsent_data = np.load(sinsent_src)
    with open(short_info_path, 'r') as f:
        short_info = json.load(f)
    with open(sinsent_info_path, 'r') as f:
        sinsent_info = json.load(f)

    seq_ids = [sinsent_info['sequence_id_list'][i] for i in sample_idx]
    ins_ids = [sinsent_info['instance_id_list'][i] for i in sample_idx]

    sinsent_vec = list(sinsent_data[sample_idx, :])  # list of np array

    mask = []
    mat = []
    for i in xrange(len(sample_idx)):
        ss = sinsent_vec[i]
        seq_id = seq_ids[i]
        ins_id = ins_ids[i]
        seq_inds = [j for j, item in enumerate(short_info['sequence_id_list'])
                    if item == seq_id]
        ins_inds = [j for j, item in enumerate(short_info['instance_id_list'])
                    if item == ins_id]
        ind = list(set(seq_inds) & set(ins_inds))
        short = list(short_data[ind, :])

        mat += ([ss] + short)
        mask += [i] * (len(short) + 1)
    # pdb.set_trace()
    return (mat, mask)

def tsne_viz(
        mat=None,
        mask=None,
        indices=None,
        output_filename=None,
        figheight=40,
        figwidth=50,
        display_progress=False):

    num_samples = len(set(mask))

    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
    colors = [colormap(np.linspace(0, 1, num_samples)[i]) for i in mask]
    # colors = [colormap(i) for i in np.linspace(0, 1, len(mat))]

    _mask = [None] + mask
    marker_tmp = [1 if _mask[i] != _mask[i-1] else 0 for i in xrange(1, len(_mask))]
    markers = ['^' if marker_tmp[i] == 1 else 'o' for i in xrange(len(marker_tmp))]
    # markers = ['*'] * len(mat)
    marker_sizes = [20*4**2 if marker_tmp[i] == 1 else 20*4**1 for i in xrange(len(marker_tmp))]
    # marker_sizes = [20*4**2] * len(mat)

    temp = sys.stdout
    if not display_progress:
        # Redirect stdout so that tsne doesn't fill the screen with its iteration info:
        f = open(os.devnull, 'w')
        sys.stdout = f
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
        ax.scatter(_x, _y, s=_s, c=_c, marker=_marker)

    # if output_filename:
    plt.savefig(output_filename, bbox_inches='tight')
    print "Image saved."
    # else:
    plt.show()


if __name__ == "__main__":
    src_dir='skipgram.txt'
    # wv = build(src_dir, delimiter=' ', header=True, quoting=csv.QUOTE_NONE)
    wv = build_tacos(short_desc_path, sinsent_desc_path, short_info_path, sinsent_info_path)
    tsne_viz(mat=np.array(wv[0]), mask=wv[1], output_filename=dir_save_im)
