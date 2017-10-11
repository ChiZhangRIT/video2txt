import os
import sys
import csv
import numpy as np
from tsne import tsne
import matplotlib.pyplot as plt
import pdb

def build(src_filename, delimiter = ',', header = True, quoting = csv.QUOTE_MINIMAL):
    fp=open(src_filename,'r')
    mat = []
    rownames = []
    for line in fp:
        word_info=line.split()
     	rownames.append(word_info[0])
     	mat.append(np.array(map(float, word_info[1: ])))

    return (mat, rownames)

def tsne_viz(
        mat=None,
        rownames=None,
        indices=None,
        colors=None,
        output_filename=None,
        figheight=40,
        figwidth=50,
        display_progress=False):
    if not colors:
        colors = ['black' for i in range(len(mat))]
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
    vocab = np.array(rownames)[indices]
    xvals = tsnemat[indices, 0]  # projected x-values
    yvals = tsnemat[indices, 1]  # projected y-values
    # Plotting:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(100)
    fig.set_figwidth(500)
    ax.plot(xvals, yvals, marker='o', linestyle='')
    # Text labels:
    for word, x, y, color in zip(vocab, xvals, yvals, colors):
        # ax.annotate(word, (x, y), fontsize=8, color=color)
        ax.annotate('', (x, y), fontsize=8, color=color)
    if output_filename:
        print "Output:"
        plt.savefig(output_filename, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
	src_dir='skipgram.txt'
	wv = build(src_dir, delimiter=' ', header=True, quoting=csv.QUOTE_NONE)
	tsne_viz(mat=np.array(wv[0]),rownames=wv[1],output_filename=None)
