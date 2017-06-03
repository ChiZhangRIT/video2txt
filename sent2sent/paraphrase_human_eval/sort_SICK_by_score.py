import pdb
from operator import itemgetter

source_file = '../data/SICK/SICK.txt'

with open(source_file, 'r') as f:
    lines = f.readlines()

# lines[0]:
        #    pair_ID
        #    sentence_A
        #    sentence_B
        #    entailment_label
        #    relatedness_score
        #    entailment_AB
        #    entailment_BA
        #    sentence_A_original
        #    sentence_B_original
        #    sentence_A_dataset
        #    entence_B_dataset
        #    SemEval_set
del lines[0]
new_data = []
for line in lines:
    temp = line.split('\t')
    new_line = []
    new_line.append(temp[0])
    new_line.append(temp[1])
    new_line.append(temp[2])
    new_line.append(float(temp[4]))
    new_line.append('\n')
    new_data.append(new_line)

sorted_data = sorted(new_data, key=itemgetter(3), reverse=True)

with open('sorted_SICK.txt', 'w') as f:
    for item in sorted_data:
        item[3] = str(item[3])
        item = '\t'.join(item)
        f.write(item)
