'''
This script splits the TACoS dataset into training and testing.
'''

import json
import pickle as pkl
import pdb

train_split = 'data/tacos/experimentalSetup/sequencesTrainValDishes.txt'
test_split = 'data/tacos/experimentalSetup/sequencesTest.txt'
vectors_path = 'vectors/'
sentences_path = 'sentences/'
description_types = ['detailed', 'short', 'singlesentence']

# extract video ids for training
with open(train_split, 'r') as f:
    train_ids = f.readlines()
# remove whitespace characters like `\n` at the end of each line
train_ids = [x.strip() for x in train_ids]

# extract video ids for testing
with open(test_split, 'r') as f:
    test_ids = f.readlines()
# remove whitespace characters like `\n` at the end of each line
test_ids = [x.strip() for x in test_ids]

vectors_files = [vectors_path + 'tacos_' + i + '.pkl' for i in description_types]
sentences_info = [sentences_path + 'tacos_' + i + '_info.json' for i in description_types]

train_vectors = []
test_vectors = []
for description_type in description_types:

    # initialize output lists
    train_vectors = []
    test_vectors = []
    train_info = {'instance_id_list': [], 'sequence_id_list': []}
    test_info = {'instance_id_list': [], 'sequence_id_list': []}
    train_sentences = []
    test_sentences = []

    # load vectors and corresponding info
    vectors_file = vectors_path + 'tacos_' + description_type + '.pkl'
    sentences_info_file = sentences_path + 'tacos_' + description_type + '_info.json'
    with open(vectors_file, 'r') as f:
        vectors = pkl.load(f)
    with open(sentences_info_file, 'r') as f:
        info = json.load(f)
    assert 'instance_id_list' in info.keys(), "key 'instance_id_list' cannot be found."
    assert 'sequence_id_list' in info.keys(), "key 'sequence_id_list' cannot be found."
    instance_id_list = info['instance_id_list']
    sequence_id_list = info['sequence_id_list']

    # load sentences
    with open(sentences_path + 'tacos_' + description_type + '.txt', 'r') as f:
        sentences = f.readlines()

    for i, vec in enumerate(vectors):

        instance_id = instance_id_list[i]
        sequence_id = sequence_id_list[i]

        if sequence_id in train_ids:  # training set
            train_vectors.append(vec)
            train_info['instance_id_list'].append(instance_id)
            train_info['sequence_id_list'].append(sequence_id)
            train_sentences.append(sentences[i])
        elif sequence_id in test_ids:  # test set
            test_vectors.append(vec)
            test_info['instance_id_list'].append(instance_id)
            test_info['sequence_id_list'].append(sequence_id)
            test_sentences.append(sentences[i])
        else:
            raise ValueError('Sequence id %s is in neither training nor test lists.' % (vec))

    # save training to disk
    with open(vectors_path + 'tacos_' + description_type + '_train.pkl', 'w') as f:
        pkl.dump(train_vectors, f)
    with open(sentences_path + 'tacos_' + description_type + '_info_train.json', 'w') as f:
        json.dump(train_info, f)
    with open(sentences_path + 'tacos_' + description_type + '_train.txt', 'w') as f:
        f.writelines(train_sentences)
    # save test to disk
    with open(vectors_path + 'tacos_' + description_type + '_test.pkl', 'w') as f:
        pkl.dump(test_vectors, f)
    with open(sentences_path + 'tacos_' + description_type + '_info_test.json', 'w') as f:
        json.dump(test_info, f)
    with open(sentences_path + 'tacos_' + description_type + '_test.txt', 'w') as f:
        f.writelines(test_sentences)

pass
