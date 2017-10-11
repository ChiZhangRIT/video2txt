from __future__ import division
import sys, keras.backend as K
import cPickle as pkl
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, TimeDistributed, Masking
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adadelta
from models.decoder import GRU_Decoder
from models.encoder import BA_LSTM
from datasets import MVAD, MPII_MD, Youtube2Text

b_s = 128
dataset = MPII_MD()

def model(video, caption):
    embedding_layer = Embedding(dataset.vocabulary_size, dataset.embedding_size, input_length=dataset.max_caption_len, mask_zero=True)
    video_embedding = TimeDistributed(Dense(dataset.embedding_size))(Masking()(Dropout(0.5)(video)))
    video_encoding_and_shots = BA_LSTM(1024, first=True, return_sequences=True)(video_embedding)
    video_encoding_and_shots2 = BA_LSTM(1024, return_sequences=False, disable_shots=True)(video_encoding_and_shots)
    caption_encoding = Dropout(0.5)(embedding_layer(caption))
    decoder = GRU_Decoder(1024, dataset.embedding_size)([video_encoding_and_shots2, caption_encoding])
    out = TimeDistributed(Dense(dataset.vocabulary_size, activation='softmax'))(Dropout(0.5)(decoder))
    return out

def loss(y_true, y_pred):
    selettore = K.clip(y_true[:, 1:, 0], 0, 1)
    return K.sum(selettore * K.sparse_categorical_crossentropy(y_pred[:,:dataset.max_caption_len-1], y_true[:,1:,0])) / K.sum(selettore)

def sparse_categorical_accuracy(y_true, y_pred):
    selettore = K.clip(y_true[:, 1:, 0], 0, 1)
    return K.sum(selettore*K.equal(y_true[:, 1:, 0],
                          K.cast(K.argmax(y_pred[:,:dataset.max_caption_len-1], axis=-1), K.floatx()))) / K.sum(selettore)

video = Input((dataset.max_video_len, 2048+4096))
caption = Input((dataset.max_caption_len, ), dtype='int32')
m = Model(input=[video, caption], output=model(video, caption))
m.compile(optimizer=Adadelta(.1), loss=loss)
exp_name = 'experiments/hmlstm_mvad_1l_noatt_s_lr.1'
print exp_name

# Training
if False:
    history = m.fit_generator(dataset.generator(b_s, 'train'), dataset.nb_train_samples, 100, validation_data=dataset.generator(b_s, 'val'), nb_val_samples=dataset.nb_val_samples, callbacks=[EarlyStopping(patience=5), ModelCheckpoint(exp_name + '_.{epoch:02d}-{val_loss:.2f}.pkl', save_best_only=True)])
    pkl.dump(history.history, open(exp_name+'_history.pkl', 'wb'))
    m.save_weights(exp_name+'.pkl')

# Test
if True:
    word_index = dataset.tokenizer.word_index
    gen = dataset.generator(b_s, 'test', show_ids=True)
    m.load_weights('data/mpiimd_model.pkl')
    seen = 0
    hypothesis_for_coco = []
    references_for_coco = []
    images_for_coco = []

    while seen < dataset.nb_test_samples:
        [X_video, X_caption], Y_gt, snippet_ids = gen.next()
        Y_t = m.predict_on_batch([X_video, X_caption])
        Y_t = np.argmax(Y_t, axis=-1)

        for t in range(dataset.max_caption_len - 1):
            X_caption[:, t + 1] = Y_t[:, t]
            if all(w == 'stoptoken' for w in
                   [word_index.keys()[word_index.values().index(Yi)] for Yi in X_caption[:, t + 1]]):
                break
            Y_t = m.predict_on_batch([X_video, X_caption])
            Y_t = np.argmax(Y_t, axis=-1)

        for pred, gt, snippet_id in zip(X_caption, Y_gt, snippet_ids):
            gt_caption = []
            pred_caption = []
            for gt_t in gt:
                gt_word = word_index.keys()[word_index.values().index(gt_t[0])]
                gt_caption.append(gt_word)
                if gt_word == 'stoptoken':
                    break
            for pred_t in pred:
                pred_word = word_index.keys()[word_index.values().index(pred_t)]
                pred_caption.append(pred_word)
                if pred_word == 'stoptoken':
                    break

            gt_caption = gt_caption[1:-1]
            pred_caption = pred_caption[1:-1]

            if not any(h for h in hypothesis_for_coco if h['image_id'] == snippet_id):
                images_for_coco.append({"id": snippet_id, "url": "", "file_name": ""})
                hypothesis_for_coco.append({"image_id": snippet_id, "id": seen, "caption": ' '.join([c.decode('utf-8').encode('ascii','ignore') for c in pred_caption])})
            references_for_coco.append({"image_id": snippet_id, "id": seen, "caption": ' '.join([c.decode('utf-8').encode('ascii','ignore') for c in gt_caption])})

            seen += 1

            print '%d / %d - %s GT: %s' % (seen, dataset.nb_test_samples, snippet_id, ' '.join(gt_caption))
            print '%d / %d - %s PR: %s' % (seen, dataset.nb_test_samples, snippet_id, ' '.join(pred_caption))

    # Evaluation
    print exp_name
    import json
    json.dump(hypothesis_for_coco, open("%s_hypothesis.json" % exp_name, 'w'))
    json.dump({'images': images_for_coco, 'annotations': references_for_coco, 'type': 'captions', 'info': {}, 'licenses': []}, open("%s_references.json" % exp_name, 'w'))

    sys.path.append('coco_caption')
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    coco = COCO("%s_references.json" % exp_name)
    cocoRes = coco.loadRes("%s_hypothesis.json" % exp_name)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        print '%s: %.3f' % (metric, score)
