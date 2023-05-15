# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.7.13 (default, Mar 28 2022, 08:03:21) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: /data/jindeng/spatio-temporal forecasting/ST-Norm/models/TCN.py
# Compiled at: 2021-08-18 11:42:09
# Size of source mod 2**32: 6281 bytes
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from scipy.linalg import block_diag
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
from models.modules import *


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class DilateConv(nn.Module):

    def __init__(self, num_nodes, n_his, n_inputs, n_outputs, kernel_size, stride, dilation, padding, tnorm_bool, snorm_bool):
        super(DilateConv, self).__init__()
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs
        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool
        if tnorm_bool:
            self.tn = TNorm(num_nodes, n_inputs)
        if snorm_bool:
            self.sn = SNorm(n_inputs)
        num = int(tnorm_bool) + int(snorm_bool) + 1
        self.conv = weight_norm(nn.Conv2d((n_inputs * num), n_outputs, kernel_size=(1, kernel_size), stride=stride, padding=(0, padding), dilation=dilation))

    def forward(self, x):
        b, c, n, t = x.shape
        x_list = [x]
        if self.tnorm_bool:
            x_tnorm = self.tn(x)
            x_list.append(x_tnorm)
        if self.snorm_bool:
            x_snorm = self.sn(x)
            x_list.append(x_snorm)
        x = torch.cat(x_list, dim=1)
        out = self.conv(x)
        return out


class TemporalBlock(nn.Module):

    def __init__(self, num_nodes, n_his, n_inputs, n_outputs, kernel_size, stride, dilation, padding, tnorm_bool, snorm_bool, dropout=0):
        super(TemporalBlock, self).__init__()
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv1 = DilateConv(num_nodes, n_his, n_inputs, n_outputs, kernel_size, stride, dilation, padding, tnorm_bool=tnorm_bool, snorm_bool=snorm_bool)
        self.conv2 = DilateConv(num_nodes, n_his, n_outputs, n_outputs, kernel_size, stride, dilation, padding, tnorm_bool=tnorm_bool, snorm_bool=snorm_bool)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net1 = nn.Sequential(self.chomp1, self.relu1, self.dropout1)
        self.net2 = nn.Sequential(self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        b, n, ic, t = x.shape
        out = self.conv1(x)
        out = self.net1(out)
        out = self.conv2(out)
        out = self.net2(out)
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)
        out_np = out.data.cpu().numpy()
        return out


class TemporalConvNet(nn.Module):

    def __init__(self, 
                    num_nodes, 
                    in_channels, 
                    n_his, 
                    n_pred, 
                    hidden_channels, 
                    n_layers, 
                    st1,
                    st2,
                    STNorm_n,
                    TSNorm_n,
                    attention_bool,
                    kernel_size=2):
        super(TemporalConvNet, self).__init__()
        layers = []
        decode_layers = []
        channels = [in_channels] + [hidden_channels] * n_layers
        self.st1 = st1
        self.STNorm_n = STNorm_n
        self.TSNorm_n = TSNorm_n
        self.multiattention = nn.ModuleList()
        if st1:
            self.multiattention.append(multiattention(self.TSNorm_n, 
            self.STNorm_n, 
            in_dim = in_channels, 
            channels = hidden_channels, 
            n = num_nodes, 
            input_size = n_his,  
            attention_bool = attention_bool))
            channels[0] = (self.TSNorm_n + self.STNorm_n)* 5
        
        if st2:
            snorm_bool = tnorm_bool = True
        else:
            snorm_bool = tnorm_bool = False
        
        for i in range(n_layers):
            dilation_size = 2 ** i
            in_channels = channels[i]
            out_channels = channels[i + 1]
            layers += [TemporalBlock(num_nodes, n_his, in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=((kernel_size - 1) * dilation_size), snorm_bool=snorm_bool, tnorm_bool=tnorm_bool)]
        self.layers = nn.ModuleList(layers)
        self.out_conv = nn.Conv2d(hidden_channels, n_pred, 1)

    def forward(self, x):
        b, t, n, c = x.size()
        x = x.permute(0, 3, 2, 1)
        if self.st1:
            x = self.multiattention[0](x)
            if not self.training:
                atten_4_part = []
                for ii  in range(4):
                    part = x[:, [k * 5 + ii for k in range(self.STNorm_n + self.TSNorm_n)], :, 0]
                    atten_4_part.append(part)
                atten_4_part = torch.cat(atten_4_part, dim=1)
                
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        else:
            out = out[..., -1:]
            out = self.out_conv(out)
        if not self.training and self.st1:
            return out, atten_4_part
        elif not self.training:
            return out, []
        else:
            return out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(name)
                print(param.shape)