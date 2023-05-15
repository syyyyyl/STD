# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import time
import csv
import numpy as np
import math
import time
from models.Wavenet import Wavenet
from models.modules import *
from models.TCN import TemporalConvNet as TCN
from models.Transformer import Transformer 

from utils.data_utils import *
from utils.math_utils import *
from utils.tester import model_inference
from utils.training import *
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import argparse
os.getcwd()
torch.backends.cudnn.benchmark = True


batch_size = 4  # batch size
test_batch_size = 48

lr = 0.0001  # learning rate

parser = argparse.ArgumentParser()
parser.add_argument('--version', type = int, default=0)
parser.add_argument('--model', type = int, default=0)
parser.add_argument('--mode', type = str, default='train')
parser.add_argument('--n_his', type = int, default=16)
parser.add_argument('--n_pred', type = int, default=3)
parser.add_argument('--n_layers', type = int, default = 4)
parser.add_argument('--hidden_channels', type = int, default = 16)
parser.add_argument('--STNorm_n', type = int, default=1)
parser.add_argument('--TSNorm_n', type = int, default=1)
parser.add_argument('--st1', type = int, default=1)
parser.add_argument('--st2', type = int, default=1)
parser.add_argument('--n_train', type = int, default = 132)
parser.add_argument('--n_val', type = int, default = 24)
parser.add_argument('--n_test', type = int, default = 24)
parser.add_argument('--n', type = int, default = 50)
parser.add_argument('--attention', type = int, default = 1)
parser.add_argument('--n_slots', type = int, default = 24)
parser.add_argument('--filename', type = str, default="BikeNYC")
parser.add_argument('--real_data', type = int, default=1)

args = parser.parse_args()
n_his = args.n_his
n_pred = args.n_pred
n_layers = args.n_layers
hidden_channels = args.hidden_channels
version = args.version
st1 = args.st1
st2 = args.st2
STNorm_n = args.STNorm_n
TSNorm_n = args.TSNorm_n
n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
n = args.n
attention = args.attention
n_slots = args.n_slots
model_name = args.model

seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # multi gpu
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
if args.model == 0:
    model_name = 'WaveNet'
elif args.model == 1:
    model_name = 'TCN'
else:
    model_name = 'Transformer'
print("st1 = ", st1)
print("st2 = ", st2)
print("attention = ", attention)
print("model ", model_name)



def main():
        print("loading data...")
        print(args.filename)
        datafile = "/public/home/tianting/ST-Norm/ST-Norm-master-multi/ST-Norm-master/data/" + args.filename + ".csv"
        print(datafile)
        dataset = data_gen(datafile, (n_train, n_val, n_test), n, n_his + n_pred, n_slots)
        if not args.real_data:
            data_return = pd.read_csv('/public/home/tianting/ST-Norm/ST-Norm-master-multi/ST-Norm-master/data/' + args.filename + '_datareturn.csv')
            data_return = np.array(data_return)
        else:
            data_return = []
        print('=' * 10)
        print("compiling model...")
        
        print('=' * 10)
        print("init model...")
                
        trainer = train(n, STNorm_n, TSNorm_n, st1, st2, attention, args.filename, n_layers, n_his, n_pred, data_return, dataset, model_name = model_name, real_data = args.real_data)
        torch.cuda.empty_cache()
        if model_name == 'WaveNet':
                model = Wavenet('cuda:0', 
                  n = n, 
                  STNorm_n = STNorm_n, 
                  TSNorm_n = TSNorm_n, 
                  in_dim = 1,
                  out_dim = n_pred, 
                  channels = hidden_channels, 
                  kernel_size = 2, 
                  blocks = 1, 
                  layers = n_layers,
                  st1 = st1,
                  st2 = st2,
                  attention_bool = attention).cuda()
        if model_name == 'TCN':
                model = TCN(num_nodes = n, 
                    in_channels = 1, 
                    n_his = n_his, 
                    n_pred = n_pred, 
                    hidden_channels = hidden_channels, 
                    n_layers = n_layers, 
                    st1 = st1,
                    st2 = st2,
                    STNorm_n = STNorm_n,
                    TSNorm_n = TSNorm_n,
                    attention_bool = attention).cuda()
        if model_name == 'Transformer':
                model = Transformer(num_nodes = n, 
                    in_channels = 1, 
                    n_his = n_his, 
                    n_pred = n_pred, 
                    hidden_channels = hidden_channels, 
                    n_layers = n_layers, 
                    st1 = st1,
                    st2 = st2,
                    STNorm_n = STNorm_n,
                    TSNorm_n = TSNorm_n,
                    attention_bool = attention).cuda()
        start = datetime.datetime.now()
        trainer.train(model, 200, new_training = True)
        end1 = datetime.datetime.now()
        print("total training time: ", end1 - start)
        trainer.eval(model)
        end2 = datetime.datetime.now()
        print("total testing time: ", end2 - end1)

if __name__ == '__main__':
    main()
