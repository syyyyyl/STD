import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from models.modules import *

class Wavenet(nn.Module):
    def __init__(self, device, STNorm_n, TSNorm_n, n,  lh_first_bool = True, in_dim=1, out_dim=12, channels=16, kernel_size=2, blocks = 1, layers=4, requires_grad = True, st1 = True, st2 = True, attention_bool = 1):
        super(Wavenet, self).__init__()
        self.blocks = blocks
        self.layers = layers
        self.st1 = st1
        self.st2 = st2
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.requires_grad = requires_grad
        self.single_attention = nn.ModuleList()
        self.multiattention = nn.ModuleList()
        
        self.STNorm_n = STNorm_n
        self.TSNorm_n = TSNorm_n
        
        self.sn = nn.ModuleList()
        
        self.tn = nn.ModuleList()
        
        self.attention_bool = attention_bool
        
        
        self.dropout = nn.Dropout(0.2)
        
        if st1:
            self.multiattention.append(multiattention(STNorm_n, 
            TSNorm_n, 
            in_dim, 
            channels, 
            n = n, 
            input_size = 16,  
            attention_bool = self.attention_bool))
            
            self.start_conv = nn.Conv2d(in_channels=(STNorm_n + TSNorm_n) * 5,
                                    out_channels = channels,
                                    kernel_size = (1,1))
        else:
            print
            self.start_conv = nn.Conv2d(in_channels = in_dim,
                                    out_channels = channels,
                                    kernel_size = (1,1))
        num = 1
        receptive_field = 1
        self.dilation = []
        for b in range(blocks):
            new_dilation = 1
            additional_scope = kernel_size - 1
            for i in range(layers):
                if st2: 
                    self.tn.append(TNorm(n, channels))
                    self.sn.append(SNorm(channels))
                    num = 3
                self.filter_convs.append(nn.Conv2d(in_channels = num * channels,
                                                   out_channels = channels,
                                                   kernel_size=(1,kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels = num * channels,
                                                 out_channels=channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                
                self.dilation.append(new_dilation)

                    # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels = channels,
                                                     out_channels = channels,
                                                     kernel_size = (1, 1)))

                    # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels = channels,
                                                 out_channels = channels,
                                                 kernel_size = (1, 1)))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2


        self.end_conv_1 = nn.Conv2d(in_channels=channels,
                                  out_channels=channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
    
        try:
            input = input.permute(0, 3, 2, 1)
            
        except:
            input = input[0]
            input = input.permute(0, 3, 2, 1)
        
        if self.st1:
            input = self.multiattention[0](input)
            if not self.training:
                atten_4_part = []
                for ii  in range(4):
                    part = input[:, [k * 5 + ii for k in range(self.STNorm_n + self.TSNorm_n)], :, 0]
                    atten_4_part.append(part)
                atten_4_part = torch.cat(atten_4_part, dim=1)
        
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        
        x = self.start_conv(x)
        skip = 0
        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            x_list = []
            x_list.append(x)
            if self.st2:
                b, c, n, t = x.shape
                
                x_tnorm = self.tn[i](x)
                x_list.append(x_tnorm)
                
                x_snorm = self.sn[i](x)
                x_list.append(x_snorm)

                # dilated convolution
            x = torch.cat(x_list, dim=1)
            filter = self.filter_convs[i](x)
            b, c, n, t = filter.shape
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
              skip = skip[:, :, :,  -s.size(3):]
            except:
              skip = 0
            skip = s + skip
            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            
        x = F.relu(skip)
        
        rep = F.relu(self.end_conv_1(x))
        
        out = self.end_conv_2(rep)
        
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
                