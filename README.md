# STAD
This is a implementation of ST-Norm. Space-temporal Attention Decomposition (STAD): An Interpretable Head Space-temporal Decomposition  Module for Multivariate Time Series

# Requirements
Python 3.7  
Numpy >= 1.17.4  
Pandas >= 1.0.3  
Pytorch >= 1.4.0

### Arguments
model: backbone architecture (default=0, 0 for wavenet /1 for tcn /2 for transformer).  
n_his: number of input steps.
hidden_channels: number of hidden channels.  
n_his: number of input steps.  
n_layers: number of hidden layers.
st1: whether to use ST(A)D.
st2: whether to use ST-Norm.
STNorm_n: number of STD.
TSNorm_n: number of TSD.
n_train: number of training batch size.
n_val: number of validating batch size.
n_test: number of testing batch size.
n: number of locations.
attention: whether to add attention mechanism to the model.
n_slots: number of slots.
filename: dataset name.
real_data: whether the data is real data or simulated data.

## Model Training and Evaluating For WaveNet with STD
```
python main.py --st1 1 --st2 0 --attention 0
```

## Model Training and Evaluating For WaveNet with STAD
```
python main.py --st1 1 --st2 0 --attention 1
```
