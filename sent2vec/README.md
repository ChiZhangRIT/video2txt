## Sentence-to-sentence Model

This work is based on easy_seq2seq. Original code can be found here: https://github.com/suriyadeepan/easy_seq2seq

### Data preparation

```
mkdir sentences  # input to sent2vec
mkdir model
```
Put the model and sentences data in *model/* and *sentences/* directory.

For TACoS dataset, run *parse_tacos.py* to generate sentences text file and corresponding info file. The generated text file includes all sentences in the entire dataset, with each line a single sentence. The info file contains lists of sequence id and detail level for each sentence. Please look at [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/vision-and-language/tacos-multi-level-corpus/) for more details about TACoS.


### Generating vectors from sentences

Edit *seq2seq.ini* file to set *mode = eval*. Specify input and output directory, then run *execute.py*.
```
# input to sen2vec. <detail_level> can be "detailed", "short" or "singlesentence"
sentences_file = sentences/tacos_<detail_level>.txt
# output vector file. <detail_level> can be "detailed", "short" or "singlesentence"
vector_file = vectors/tacos_<detail_level>.pkl
```

Note: To assgin a GPU device, apply
```
export CUDA_VISIBLE_DEVICES="0"
```
