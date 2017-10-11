## Vector-to-sentence Model

This work is based on easy_seq2seq. Original code can be found [here](https://github.com/suriyadeepan/easy_seq2seq).

### Data preparation

Create directory
```
mkdir sentences
mkdir vectors
```
Move the sentences files and corresponding info files under sentences directory.
Move the vectors files under vectors directory.

### Training

Edit *seq2seq.ini* file to set *mode = train*.
```
python execute.py
```
Note: To assgin a GPU device, use
```
export CUDA_VISIBLE_DEVICES="0"
```

### Evaluation

Edit *seq2seq.ini* file to set *mode = eval*
```
python execute.py
```

### TensorBoard

Run a TensorBoard server in a separate process for real-time monitoring of training progress and evaluation metrics.
```
tensorboard --logdir=log_dir/ --port=2941
```
