## Sentence-to-sentence Model

This work is based on easy_seq2seq. Original code can be found here: https://github.com/suriyadeepan/easy_seq2seq

### Data preparation

Put the sentences data in *sentences/* directory
```
mkdir sentences
mkdir model
```

### Generating vectors from sentences

Edit *seq2seq.ini* file to set *mode = eval*. Specify input and output directory, then run
```
python execute.py
```

Note: To assgin a GPU device, use
```
export CUDA_VISIBLE_DEVICES="0"
```
