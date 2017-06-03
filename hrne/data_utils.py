import os
import numpy as np
import json
import pickle as pkl
import copy
import pdb
from tqdm import *


def bucketing(vec_count, buckets, vec_dim, curr_sample):
    ''' throw a sample into corresponding bucket.

    Args:
    vec_count: count the number of vectors in current sample.
    buckets: bucket list dealing with variation of number of sentences.
    vec_dim: dimension of sentence vectors.
    curr_sample: current sample, must be an numpy array.

    Returns:
    bucket_id: id of bucket for current sample.
    curr_sample: updated current sample.
    '''
    try:
        bucket_id = min([i for i in xrange(len(buckets))
                         if vec_count <= buckets[i]])
        # padding 0 vectors if num of sentences is less than current bucket
        num_pad = buckets[bucket_id] - vec_count
        pad = np.zeros((num_pad, vec_dim))
        curr_sample = np.vstack((curr_sample, pad))  # shape (buckets[bucket_id], vec_dim)
    except:
        bucket_id = len(buckets) - 1
        # chop off extra sentences
        curr_sample = curr_sample[:buckets[-1], :]
    return bucket_id, curr_sample

def load_vectors(vec_file, info_file, buckets):
    """ load data into buckets.

    Args:
    vec_file: a numpy file containing the vector representation of sentences.
    info_file: an infomation file specifying the sequence and instance ids of
               the sentences in vec_file.
    buckets: bucket list dealing with variation of number of sentences. If buckets are not changed during training, this function is called only once.

    Returns:
    data: a dict of buckets (as lists) of encoder inputs. data[bucket_id] has shape of (num_sample_in_this_bucket, bucket[bucket_id], vec_dim)
    seq_ids: a dict of buckets (as lists) of sequence ids.
    ins_ids: a dict of buckets (as lists) of sequence ids.

    Raises:
    ValueError: if length of seq id and ins id are not identical.
    ValueError: if vec_file does not correspond to info_file.
    """
    vectors = np.load(vec_file)
    with open(info_file, 'r') as f:
        info = json.load(f)

    if len(info['sequence_id_list']) != len(info['instance_id_list']):
        raise ValueError('Invalid info file: len(info[''sequence_id_list'']) should be equal to len(info[''instance_id_list'']) but got %d != %d' % (len(info['sequence_id_list']), len(info['instance_id_list'])))
    if vectors.shape[0] != len(info['instance_id_list']):
        raise ValueError('Number of vectors and length of info list are not identical. %d != %d' % (len(info['sequence_id_list']), len(info['instance_id_list'])))

    num_vec, vec_dim = vectors.shape
    prev_seq_id, prev_ins_id = None, None
    seq_ids = {i: [] for i in xrange(len(buckets))}
    ins_ids = {i: [] for i in xrange(len(buckets))}
    data = {i: [] for i in xrange(len(buckets))}

    for vec_idx in tqdm(xrange(num_vec)):
        curr_seq_id, curr_ins_id = info['sequence_id_list'][vec_idx], info['instance_id_list'][vec_idx]
        if curr_seq_id != prev_seq_id or curr_ins_id != prev_ins_id:
            if prev_seq_id is not None:
                bucket_id, curr_sample = bucketing(vec_count, buckets, vec_dim, curr_sample)
                seq_ids[bucket_id].append(prev_seq_id)
                ins_ids[bucket_id].append(prev_ins_id)
                data[bucket_id].append(curr_sample)
            vec_count = 1
            curr_sample = vectors[vec_idx]
        else:
            vec_count += 1
            curr_sample = np.vstack((curr_sample, vectors[vec_idx]))
        prev_seq_id = curr_seq_id
        prev_ins_id = curr_ins_id

    # throw the last sample into bucket
    bucket_id, curr_sample = bucketing(vec_count, buckets, vec_dim, curr_sample)
    seq_ids[bucket_id].append(prev_seq_id)
    ins_ids[bucket_id].append(prev_ins_id)
    data[bucket_id].append(curr_sample)

    return data, seq_ids, ins_ids

def load_data(filename, buckets, vec_file, info_file):
    """ load encoder_inputs and ground truths.
    Encoder_inputs is loaded from vector file, then saved to local after
    bucketing. If it is already saved, loaded from disk.

    Args:
    filename: file path containing input vectors and related information. Filename usually look like <dataset_mode_buckets.pkl>, e.g., 'tacos_train_20_40.pkl'.
    buckets: bucket list dealing with variation of number of sentences.
    vec_file: detailed description vector file.
    info_file: detailed description info file. Must be correspond to vec_file.

    Returns:
    encoder_inputs: a dict of buckets (as lists) of encoder inputs. data[bucket_id] has length of num_sample_in_this_bucket, each with shape (bucket[bucket_id], vec_dim).
    seq_ids: a dict of buckets (as lists) of sequence ids.
    ins_ids: a dict of buckets (as lists) of sequence ids.
    """
    if os.path.exists(filename):
        # load data from existing metadata.
        print "Loading vectors from existing data ..."
        with open(filename, 'r') as f:
            metadata = pkl.load(f)
        encoder_inputs = metadata['data']
        seq_ids = metadata['seq_ids']
        ins_ids = metadata['ins_ids']
        print "vector file is loaded from", metadata['src_file']
        print "buckets are", metadata['buckets']
        if metadata['buckets'] != buckets:
            raise ValueError("Buckets are not identical.")
    else:
        # load data from vec_file and bucketing vectors.
        print "Bucketing vectors ..."
        encoder_inputs, seq_ids, ins_ids = load_vectors(vec_file, info_file, buckets)
        # save data to local
        metadata = {'data': encoder_inputs, 'seq_ids': seq_ids, 'ins_ids': ins_ids, 'src_file': vec_file, 'buckets': buckets}
        with open(filename, 'w') as f:
            pkl.dump(metadata, f)
    del metadata

    return encoder_inputs, seq_ids, ins_ids

def correspond_gt(encoder_inputs, seq_ids, ins_ids, gt_sent_file, gt_info_file, buckets, description_type='singlesentence'):
    """ find corresponding ground truth sentences for TACoS dataset, given sequence ids and instance ids.

    Args:
    encoder_inputs: a dict of buckets (as lists) of encoder inputs. data[bucket_id] has length of num_sample_in_this_bucket, each with shape (bucket[bucket_id], vec_dim).
    seq_ids: a dict of buckets (as lists) of sequence ids.
    ins_ids: a dict of buckets (as lists) of sequence ids.
    gt_sent_file: the sentences as ground truth.
    gt_info_file: the information file corresponding to gt_sent_file.
    buckets: a list of buckets used in bucketing.
    description_type: 'detailed', 'short', or 'singlesentence'.

    Returns:
    encoder_inputs: the samples with unmatched seq-id or ins_id are removed.
    seq_ids: the samples with unmatched seq-id or ins_id are removed.
    ins_ids: the samples with unmatched seq-id or ins_id are removed.
    gt: a dict of buckets (as lists) of ground truth sentences.

    Raises:
    ValueError: if description_type is not one of 'detailed', 'short', or 'singlesentence'.
    ValueError: if find more than one seq/ins matches between detailed and singlesentence description.
    """
    if description_type == 'detailed':
        pass
    elif description_type == 'short':
        pass
    elif description_type == 'singlesentence':
        seq_ids_copy = copy.deepcopy(seq_ids)
        ins_ids_copy = copy.deepcopy(ins_ids)

        with open(gt_sent_file, 'r') as f:
            sentences = f.readlines()

        with open(gt_info_file, 'r') as f:
            ss_info = json.load(f)
        ss_seq_ids = ss_info['sequence_id_list']
        ss_ins_ids = ss_info['instance_id_list']

        print "Extracting and formatting ground truth sentences..."
        gt = {i: [] for i in xrange(len(buckets))}
        for bucket_id in trange(len(buckets), desc='  buckets'):
            for i in reversed(trange(len(seq_ids[bucket_id]), desc='sentences')):
                detailed_seq_id = seq_ids[bucket_id][i]
                detailed_ins_id = ins_ids[bucket_id][i]
                seq_inds = [j for j, x in enumerate(ss_seq_ids) if x == detailed_seq_id]
                ins_inds = [j for j, x in enumerate(ss_ins_ids) if x == detailed_ins_id]
                # find intersection of matched seq id and ins id
                ind = list(set(seq_inds) & set(ins_inds))
                if len(ind) == 0:
                    # if ind is empty, then this sample does not exist in singlesentence.
                    # remove this sample in encoder_inputs
                    del seq_ids_copy[bucket_id][i]
                    del ins_ids_copy[bucket_id][i]
                    del encoder_inputs[bucket_id][i]
                elif len(ind) == 1:
                    # only 1 sample match, put singlesentence into gt
                    gt[bucket_id].append(sentences[ind[0]].rstrip())
                    # gt shares seq/ins_ids_copy with encoder_inputs now.
                else:
                    raise ValueError('Find more than one seq/ins matches between detailed and singlesentence description.')
                    pdb.set_trace()
    else:
        raise ValueError('Invalid description_type.')

    return encoder_inputs, seq_ids_copy, ins_ids_copy, gt
