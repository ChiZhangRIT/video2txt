import pdb
from operator import itemgetter

sorted_SICK_file = 'sorted_SICK.txt'

def sort_by_id(result_file, save_file_path, sorted_SICK_file=sorted_SICK_file):

    # extract sorted ids
    with open(sorted_SICK_file, 'r') as f:
        lines = f.readlines()
    ids = []
    for line in lines:
        temp = line.split('\t')
        sent_id = int(temp[0])
        ids.append(sent_id)

    # assign result sentences ids
    with open(result_file, 'r') as f:
        result_sents = f.readlines()
    pdb.set_trace()
    temp = {}
    ascending_ids = sorted(ids)
    for i in xrange(len(result_sents)):
        try:
            temp[ascending_ids[i]] = result_sents[i]
        except:
            pdb.set_trace()

    sorted_sents = []
    with open(save_file_path, 'w') as f:
        for i in ids:
            f.write(str(i)+'\t'+temp[i])
            # sorted_sents.append([str(i)+'\t'+temp[i]])

    # pdb.set_trace()
    #
    # with open(save_file_path, 'w') as f:
    #     f.write(sorted_sents)



sort_by_id('../result/caption_300units_noatt_20kvocab/SICK_caption_300units_noatt_20kvocab_output.txt', 'sorted_SICK_caption_300units_noatt_20kvocab_output.txt')
print 'Done.'

sort_by_id('../result/caption_300units_noatt_50kvocab/SICK_caption_300units_noatt_50kvocab_output.txt', 'sorted_SICK_caption_300units_noatt_50kvocab_output.txt')
print 'Done.'

sort_by_id('../result/caption_1024units_noatt_50kvocab/SICK_caption_1024units_noatt_50kvocab_output.txt', 'sorted_SICK_caption_1024units_noatt_50kvocab_output.txt')
print 'Done.'
