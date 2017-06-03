# import penseur
import pdb, argparse, sys
import h5py
import nltk
import pickle as pkl
from keras.preprocessing import sequence
import numpy as np
from execute import GlobalSip
import unicodedata

reload(sys)
sys.setdefaultencoding('utf8')

path_h5file_short = '/cis/phd/cxz2081/video2txt/nyt_corpus/h5file_data/'
# path_h5file_long = '/shared/kgcoe-research/mil/sent2vec/hrne/data/NYT/nyt_corpus/nyt_corpus/h5file_data_all_summarized_articles/'
# path_to_save_feature = '/shared/kgcoe-research/mil/sent2vec/hrne/data/NYT/nyt_corpus/nyt_corpus/h5file_skipthought_selected_article/'
path_to_save_feature_short = '/cis/phd/cxz2081/video2txt/nyt_corpus/h5file_globalsip_nyt_short/'
# p = penseur.Penseur()
g = GlobalSip()
# pdb.set_trace()
# h5file keys:
# [u'abstract', u'full_text', u'lead_para', u'url', u'word_count']
YEARS_HAS_ABSTRACT = [str(i) for i in range(1996,2008)]
MAX_LEN = 40
SKIPTHOUGHT_DIM = 300

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Arguments for extracting xml ')
    parser.add_argument('--year', dest='year', help='years of nyt corpus to extract',
                        default=1987, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def extract_skipthought(path_to_h5file, year):
    data = h5py.File(path_to_h5file)
    length = len(data['abstract'])

    output_abstract = []
    output_full_text_features = np.array([])
    output_word_count = []
    output_sent_count = []
    output_url = []
    for i in xrange(length):
        abstract = nltk.sent_tokenize(unicodedata.normalize('NFKD',unicode(data['abstract'][i].replace('\n','').strip())).encode('ascii','ignore'))
        # ipdb.set_trace()
        if len(abstract) == 1:

            full_text = nltk.sent_tokenize(unicodedata.normalize('NFKD',unicode(data['full_text'][i].replace('\n','').strip())).encode('ascii','ignore'))
            sentence_count = len(full_text)
            if 10<=sentence_count<=40:
                save_full_text_feature = np.zeros((MAX_LEN, SKIPTHOUGHT_DIM))
                word_count = data['word_count'][i]
                url = data['url'][i]

                output_abstract.append(abstract[0])
                output_word_count.append(word_count)
                output_sent_count.append(sentence_count)
                output_url.append(url)
                # if output_abstract_features.size:
                    # output_abstract_features = np.vstack((output_abstract_features,p.get_vector(abstract[0]))
                # else:
                    # output_abstract_features = p.get_vector(abstract[0])

                for j in xrange(sentence_count):
                    # pdb.set_trace()
                    save_full_text_feature[j] = g.get_vector(full_text[j])
                save_full_text_feature.astype('float32')
                if output_full_text_features.size:
                    output_full_text_features = np.vstack((output_full_text_features, save_full_text_feature))
                else:
                    output_full_text_features = save_full_text_feature
        if i%100==0:
            print i
    with h5py.File(path_to_save_feature_short+year+'_globalsip_features.h5','w') as hf:
        hf.create_dataset('abstract',data = output_abstract)
        # hf.create_dataset('lead_para',data = all_lead_paragraph)
        hf.create_dataset('full_text_feat',data = output_full_text_features)
        hf.create_dataset('word_count',data = output_word_count)
        hf.create_dataset('sent_count',data = output_sent_count)
        hf.create_dataset('url',data = output_url)



def count_number_sentences(path_to_h5file, year):
    data = h5py.File(path_to_h5file)
    list_number_sentences = []
    length = len(data['abstract'])
    for i in xrange(length):
        lead_para = data['lead_para'][i].replace('\n','').strip()
        number_sent_in_lead_para = len(nltk.sent_tokenize(lead_para))
        list_number_sentences.append(number_sent_in_lead_para)
    with open(year+'_no_sent_long_leadpara.pkl','w') as f:
        pkl.dump(list_number_sentences,f)




if __name__ == '__main__':
    args = parse_args()
    year = args.year.strip()
    # path_to_extract_skipthought = path_h5file_short+'year_'+args.year+'_data.h5'
    path_to_extract_skipthought = path_h5file_short+'year_'+year+'_data.h5'
    # ipdb.set_trace()



    extract_skipthought(path_to_extract_skipthought, year)

    # count_number_sentences(path_to_extract_skipthought, args.year)
