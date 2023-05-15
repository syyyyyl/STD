
import sys, torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple
import numpy as np
from collections import defaultdict
from models.modules import *


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Attention(nn.Module):

    def __init__(self, num_nodes, 
        in_channels, 
        key_kernel_size, 
        snorm_bool, 
        tnorm_bool):
        
        super(Attention, self).__init__()
        hidden_channels = in_channels // 2
        key_padding = key_kernel_size - 1
        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool
        if tnorm_bool:
            self.tn = TNorm(num_nodes, in_channels)
        if snorm_bool:
            self.sn = SNorm(in_channels)
        num = int(tnorm_bool) + int(snorm_bool) + 1
        self.q_W = nn.Conv1d((num * in_channels), hidden_channels, key_kernel_size, 1, key_padding, bias=False)
        self.k_W = nn.Conv1d((num * in_channels), hidden_channels, key_kernel_size, 1, key_padding, bias=False)
        self.v_W = nn.Conv1d((num * in_channels), hidden_channels, 1, bias=False)
        self.o_W = nn.Conv1d(hidden_channels, in_channels, 1, bias=False)
        self.ff_W = nn.Sequential(nn.Conv1d(in_channels, in_channels, 1), nn.ReLU(), nn.Conv1d(in_channels, in_channels, 1))
        self.chomp = Chomp1d(key_padding)

    def forward(self, input):
        b, c, n, t = input.shape
        x = input
        x_list = [x]
        if self.tnorm_bool:
            x_tnorm = self.tn(x)
            x_list.append(x_tnorm)
        if self.snorm_bool:
            x_snorm = self.sn(x)
            x_list.append(x_snorm)
       
        x = torch.cat(x_list, dim=1)
        x_f = x.permute(0, 2, 1, 3).reshape(b * n, -1, t)
        input_f = input.permute(0, 2, 1, 3).reshape(b * n, -1, t)
        q = self.chomp(self.q_W(x_f))
        k = self.chomp(self.k_W(x_f))
        v = self.v_W(x_f)
        attn = torch.bmm(q.permute(0, 2, 1), k)
        upper_mask = torch.triu((torch.ones(b * n, t, t)), diagonal=1).cuda()
        attn = attn - 1000 * upper_mask
        attn = torch.softmax(attn, dim=(-1))
        attn_out = torch.bmm(attn, v.permute(0, 2, 1)).permute(0, 2, 1)
        out_f = input_f + self.o_W(attn_out)
        out_f = out_f + self.ff_W(out_f)
        out = out_f.view(b, n, -1, t).permute(0, 2, 1, 3).contiguous()
        return out


class Transformer(nn.Module):

    def __init__(self, num_nodes, 
                    in_channels, 
                    n_his, 
                    n_pred, 
                    hidden_channels, 
                    n_layers, 
                    st1,
                    st2,
                    TSNorm_n,
                    STNorm_n,
                    attention_bool,
                    atten = False,
                    n=None, ext=False, daily_slots=None, ext_channels=None):
                    
        super(Transformer, self).__init__()
        self.ext_flag = ext
        self.relu = nn.ReLU()
        self.st1 = st1
        self.TSNorm_n = TSNorm_n
        self.STNorm_n = STNorm_n
        self.multiattention = nn.ModuleList()
        num = 1
        channels = [in_channels] + [hidden_channels] * n_layers
        if st1:
            self.multiattention.append(multiattention(STNorm_n, 
            TSNorm_n, 
            in_dim = in_channels, 
            channels = hidden_channels, 
            n = num_nodes, 
            input_size = n_his,  
            attention_bool = attention_bool))
            num = (STNorm_n + TSNorm_n) * 5
        
        if st2:
            snorm_bool = tnorm_bool = True
        else:
            snorm_bool = tnorm_bool = False
            
        self.in_conv = nn.Conv2d(num * in_channels, hidden_channels, 1)
        layers = []
        for i in range(n_layers):
            layers += [Attention(num_nodes, hidden_channels, 3, snorm_bool=snorm_bool, tnorm_bool=tnorm_bool)]
        else:
            self.layers = nn.ModuleList(layers)
            self.out_conv = nn.Conv2d(hidden_channels, n_pred, 1)

    def forward(self, x):
        b, t, n, ic = x.size()
        x = x.permute(0, 3, 2, 1)
        if self.st1:
            x = self.multiattention[0](x)
            if not self.training:
                atten_4_part = []
                for ii  in range(4):
                    part = x[:, [k * 5 + ii for k in range(self.STNorm_n + self.TSNorm_n)], :, 0]
                    atten_4_part.append(part)
                atten_4_part = torch.cat(atten_4_part, dim=1)
                
        x = self.in_conv(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        else:
            out = x[..., -1:]
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
                
