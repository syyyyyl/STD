import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to("cuda:0")
        c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to("cuda:0")
        output, _ = self.lstm(input_seq)
        pred = self.linear(output)
        pred = pred[:, -1, :]
        return pred


class SN(nn.Module):
    def __init__(self, input_size, channels, output_size = 16):
        super(SN, self).__init__()
        self.lstm = LSTM(input_size, hidden_size = 1, num_layers = 1, output_size = output_size, batch_size = 8)

    def forward(self, x):
        
        x_lstm = x.permute(0, 2, 3, 1)
        
        x_ = x[:, 0, :, :]
        
        x_lstm = x_.reshape(x_lstm.shape[0], x_lstm.shape[1], x_lstm.shape[2])
        
        x_lstm = self.lstm(x_lstm)
        
        output = x_lstm
        
        output = output.reshape(output.shape[0], 1, 1, output.shape[1])
    
        out = x - x.mean(2, keepdims=True) + output
        
        return out


class TN(nn.Module):
    def __init__(self,  n, track_running_stats=True, momentum=0.1):
        super(TN, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, 1, n, 1) + torch.randn(1, 1, n, 1) * 0.01)
        self.register_buffer('running_mean', torch.zeros(1, 1, n, 1))
        self.momentum = momentum
        
    def forward(self, x):
        
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            else:
                mean = self.running_mean
        else:
            mean = x.mean((3), keepdims=True)
        out = x - mean + self.beta
        return out
        
class SNorm(nn.Module):
    def __init__(self,  channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5
        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    def __init__(self,  num_nodes, channels, track_running_stats=True, momentum=0.1):
        super(TNorm, self).__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer('running_mean', torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer('running_var', torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x):
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
                    self.running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out

    

class SELayer(nn.Module):
    
    def __init__(self, channel, reduction = 2, batch_first = True):
        super(SELayer, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.batch_first = batch_first
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, channel), bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(max(channel // reduction, channel), channel, bias = False),
            nn.Sigmoid()
            )
        self.register_buffer('attention_', torch.ones(1,channel))
        
    def forward(self, x):
       
        if not self.batch_first:
            x = x.permute(1,0,2,3)  
            
        b, c, _, _ = x.size() 
        y = self.avg_pool(x).view(b,c) #size = (batch,channel)
        
                
        y = self.fc(y).view(b,c,1,1)  #size = (batch,channel,1,1)

        self.attention_ = y.mean(0)
        
        out = x * y.expand_as(x) #size = (batch,channel,w,h)
        
        if not self.batch_first: 
            out = out.permute(1,0,2,3) #size = (channel,batch,w,h)

        return out
    
class X_minus_STN(nn.Module):
    
    def __init__(self):
        
        super(X_minus_STN, self).__init__()
        
    def forward(self, x, n):
        
        out = x - n
        
        return out
    
class single_attention(nn.Module):
    
    def __init__(self, s_first_bool, in_dim, channels, input_size, n, attention_bool, skip = True):
        
        super(single_attention, self).__init__()
        self.s_first_bool = s_first_bool
        self.skip = skip
        self.tn = nn.ModuleList()
        self.sn = nn.ModuleList()
        
        #self.attention = nn.ModuleList()
        for tn_i in range(2):
            self.tn.append(TN(n=n))
        for sn_i in range(2):
            self.sn.append(SN(channels = in_dim, input_size = input_size, output_size = input_size))
        self.minus = X_minus_STN()
        self.attention_bool = attention_bool
        if self.attention_bool:
            self.attention = SELayer(4)
    def forward(self, x):
        
        x_stnorm = []
        if self.s_first_bool == True:
        
            local = self.sn[0](x)
            
            global_ = self.minus(x, local)
        
            local_high = self.tn[0](local)
            x_stnorm.append(local_high)
        
            local_low = self.minus(local, self.tn[0](local))
            x_stnorm.append(local_low)
        
            global_high = self.tn[1](global_)
            x_stnorm.append(global_high)
        
            global_low = self.minus(global_, self.tn[1](global_))
            x_stnorm.append(global_low)
        
        else:
            high = self.tn[0](x)
            
            low = self.minus(x, high)
        
            high_local = self.sn[0](high)
            x_stnorm.append(high_local)

            high_global = self.minus(high, self.sn[0](high))
            x_stnorm.append(high_global)
            
            low_local = self.sn[1](low)
            x_stnorm.append(low_local)
            
            low_global = self.minus(low, self.sn[1](low))
            x_stnorm.append(low_global)
        x_stnorm = torch.cat(x_stnorm, dim=1)
        if self.attention_bool:
            x_stnorm = self.attention(x_stnorm)
            if self.skip:
                return torch.concat([x_stnorm, x], axis = 1)
            else:
                return x_stnorm
        else:
            if self.skip:
                return torch.concat([x_stnorm, x], axis = 1)
            else:
                return x_stnorm
    
    
class multiattention(nn.Module):
    
    def __init__(self, STNorm_n, TSNorm_n, in_dim, channels, input_size, n, attention_bool, skip = True):
        super(multiattention, self).__init__()
        self.single_attention_ = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.skip = skip
        self.STNorm_n = STNorm_n
        self.TSNorm_n = TSNorm_n
        
        self.attention_bool = attention_bool
        
        for i_t in range(STNorm_n):
            self.single_attention_.append(single_attention(s_first_bool = True, in_dim = in_dim, channels = channels, input_size = input_size, n=n, attention_bool = attention_bool, skip = self.skip))
        
        for i_f in range(TSNorm_n):
        
            self.single_attention_.append(single_attention(s_first_bool = False, in_dim = in_dim, channels = channels, input_size = input_size, n=n, attention_bool = attention_bool, skip = self.skip))
        
    def forward(self, x):
        x_out = []
        for atten in self.single_attention_:
            x_out.append(atten(x))
        x_out = torch.cat(x_out, dim=1)
        return x_out
        
class Model_light(nn.Module):
    def __init__(self, device, STNorm_n, TSNorm_n, n, in_dim=1, out_dim = 3, channels=16, attention_bool = 1):
        super(Model_light, self).__init__()
        
        self.multiattention = nn.ModuleList()
        
        self.single_attention = nn.ModuleList()

        self.STNorm_n = STNorm_n
        
        self.TSNorm_n = TSNorm_n
        
        self.attention_bool = attention_bool

        self.start_conv_ = nn.Conv2d(in_channels = (self.STNorm_n + self.TSNorm_n) * 5,
                                    out_channels= 1,
                                    kernel_size=(1,1))

        self.minus = X_minus_STN()
        
        self.dilation = []
        
        self.multiattention.append(multiattention(self.STNorm_n, 
            self.TSNorm_n, 
            in_dim, 
            channels, 
            n = n, 
            input_size = 16,  
            attention_bool = self.attention_bool))

        self.linear = nn.Linear(channels, 3)
        self.rrlu = torch.nn.RReLU()

    def forward(self, input):
        input = input.permute(0, 3, 2, 1)
        
        input = self.multiattention[0](input)
        
        in_len = input.size(3)

        x = input
        x = self.start_conv_(x)
        out = self.rrlu(self.linear(x))
        out = out.permute(0, 3, 2, 1)
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