'''
This script generates a text file containing all sentences in TACoS dataset.
'''

import os
import pdb
import pickle as pkl
import json

# file to save the vectors
output_file = "sentences/tacos_"
info_file = "sentences/tacos_"

class BreakAllTheLoops(BaseException): pass

# generate text file and info file for DETAILED description.
detail_levels = ['detailed', 'short', 'singlesentence']

for detail_level in detail_levels:
    print('Generating sentences file and info file for %s description.' % detail_level)
    sequence_id_list = []
    instance_id_list = []
    with open(output_file + detail_level + '.txt', "w") as output_txt:
        for root, _, files in os.walk("./data/tacos/" + detail_level):
            for file in files:
                if file.endswith(".txt"):

                    # file attributes looks like
                    # root: ['.', 'data', 'tacos', 'detail_level', 'sequence_id']
                    attrabutes = root.split("/")
                    sequence_id = attrabutes[-1]  # e.g., 's24-d50'
                    instance_id = file.split(".")[0]  # e.g., '1', '2', ..., '20'

                    # read lines from text files
                    with open(root+'/'+file, 'r') as input_txt:
                        sentences_orig = input_txt.readlines()
                    # add "." at the end of sentences
                    sentences = [i.split('\n')[0]+' .\n'
                                 for i in sentences_orig]

                    # document detail_level and sequence_id
                    num_sentences = len(sentences)
                    sequence_id_tmp = [sequence_id] * num_sentences
                    sequence_id_list += sequence_id_tmp
                    instance_id_tmp = [instance_id] * num_sentences
                    instance_id_list += instance_id_tmp

                    # write sentences to file
                    output_txt.writelines(sentences)

    # write corresponding sentences information to file
    info = {'sequence_id_list': sequence_id_list,
            'instance_id_list': instance_id_list}
    with open(info_file + detail_level + '_info.json', "wb") as info_json:
        json.dump(info, info_json)
