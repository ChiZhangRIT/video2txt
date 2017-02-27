## Sentence-to-sentence Model

This work is based on easy_seq2seq. Original code can be found here: https://github.com/suriyadeepan/easy_seq2seq

### Data preparation

Create directory
```
mkdir log_dir
mkdir model
```
Download original data
```
cd data/
bash pull_data.sh
```
Note that some characters in the original data is not in the right codec. We are using our own dataset.

All sentences pairs were extracted from MSCOCO + Flickr30k + MSR-VTT + MSVD.

### Training

Edit *seq2seq.ini* file to set *mode = train*. To use pre-trained embedding, set *pretrained_embedding = true*
```
python execute.py
```
Note: Set *trainable=True* in *embedding = vs.get_variable(...)* (line 762) in *embedding/rnn_cell.py* to enable training on pre-trained embedding.

### inference

Edit *seq2seq.ini* file to set *mode = test*
```
python execute.py
```

### TensorBoard

Run a TensorBoard server in a separate process for real-time monitoring of training progress and evaluation metrics.
```
tensorboard --logdir=log_dir/ --port=6364
```
