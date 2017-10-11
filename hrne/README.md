## Hierarchical Text Summarizer


### Data preparation


### Training

Edit *variables.py* file to set parameters.
```
python HRNE_SA.py
```
Note: To assgin GPU devices, use
```
export CUDA_VISIBLE_DEVICES="0,1"
```

### Evaluation


### Inference


### TensorBoard

Run a TensorBoard server in a separate process for real-time monitoring of training progress and evaluation metrics.
```
tensorboard --logdir=log_dir/ --port=2941
```
