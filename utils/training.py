from __future__ import print_function
import os
import time
import csv
import numpy as np
import math
import time
from models.Wavenet import Wavenet

from utils.data_utils import *
from utils.math_utils import *
from utils.tester import model_inference, model_inference_eval
from utils.recorder import *
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import argparse
import datetime
from torchsummary import summary
os.getcwd()
torch.backends.cudnn.benchmark = True


batch_size = 8  # batch size
test_batch_size = 24

lr = 0.0001  # learning rate

class train():
    def __init__(self, n, STNorm_n, TSNorm_n, st1, st2, attention, filename, n_layers, n_his, n_pred, data_return, dataset, model_name, real_data = 1):
        super(train, self).__init__()
        target_n = "STNorm{}_TSNorm{}_l{}_his{}_pred{}_st1{}_st2{}".format(STNorm_n, TSNorm_n, n_layers, n_his, n_pred, st1, st2) + 'attention' + str(attention) + filename + model_name
        target_fname = '{}'.format(target_n)
        self.target_model_path = os.path.join('/public/home/tianting/ST-Norm/ST-Norm-master-multi/ST-Norm-master/'+'MODEL', '{}.h5'.format(target_fname))
        self.criterion = nn.MSELoss()

        self.min_rmse = 10000000
        self.min_val = np.array([4e1, 1e5, 1e5] * 3)
        self.min_va_val = np.array([4e1, 1e5, 1e5] * 3)
        self.dataset = dataset
        self.n = n
        self.STNorm_n = STNorm_n
        self.TSNorm_n = TSNorm_n
        self.st1 = st1
        self.st2 = st2
        self.attention = attention #bool
        self.filename = filename 
        self.n_layers = n_layers
        self.n_his = n_his
        self.n_pred = n_pred
        self.data_return = data_return
        self.model_name = model_name
        self.real_data = real_data
        print('=' * 10)
        print("pretraining model...")
        print("releasing gpu memory....")


    def train_TfL(self, model1, model2, nb_epoch1 =  500, nb_epoch2 =  500):
        
        model1.train()
        stop = 0
        min_rmse = 1000000
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * 3)
        
        for p in model1.parameters():
                if p.dim() > 1:
                  nn.init.xavier_uniform_(p)
                else:
                  nn.init.uniform_(p)
                  
        for p in model2.parameters():
                if p.dim() > 1:
                  nn.init.xavier_uniform_(p)
                else:
                  nn.init.uniform_(p)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model1.parameters()), lr=lr)
        for epoch in range(nb_epoch1):  # loop over the  dataset multiple times
            model1.train()
            start = datetime.datetime.now()
            for j, x_batch in enumerate(gen_batch(self.dataset.get_data('train'), batch_size, dynamic_batch = True, shuffle = True)):
                xh = x_batch[:, 0: self.n_his]
                y = x_batch[:, self.n_his:self.n_his + self.n_pred]
                xh = torch.tensor(xh, dtype=torch.float32).cuda()
                y = torch.tensor(y, dtype=torch.float32).cuda()
                model1.zero_grad()
                pred = model1(xh)
                #print("get pred")
                loss = self.criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model1.parameters(), 10)
                optimizer.step()
         
            end1 = datetime.datetime.now()
            print("model1 training time:", end1- start)
            if epoch % 1 == 0:
            
                model1.eval()
            
                min_va_val, min_val, _, __, temp  = model_inference(model1, self.dataset, test_batch_size, self.n_his, self.n_pred, min_va_val, min_val, self.n)
            
                print(f'Epoch {epoch}:')
                va, te = min_va_val, min_val

                for i in range(self.n_pred):
                    print(f'MAPE {va[i*3]:7.3%}, {te[i*3]:7.3%};'
                        f'MAE  {va[i*3+1]:4.3f}, {te[i*3+1]:4.3f};'
                        f'RMSE {va[i*3+2]:6.3f}, {te[i*3+2]:6.3f}.')
                end2 = datetime.datetime.now()
                print("model1 testing time:", end2- end1)
                total_rmse = np.sum([va[i*3+2] for i in range(self.n_pred)])
                if total_rmse < min_rmse:
                    torch.save(model1.state_dict(), self.target_model_path)
                    min_rmse = total_rmse
                    stop = 0
                else:
                    stop += 1
                if stop == 20:
                    break
                
                #if attention:
                    #attention_path = '/public/home/tianting/ST-Norm/ST-Norm-master-multi/ST-Norm-master/' + 'output/1221/attention_track' + filename
                    #record_weights_track(epoch, model1, dataset, attention_path, lh_first_t_n, lh_first_f_n, st1, st2)
                #print(model1.state_dict()['multiattention.0.single_attention_.0.attention.0.attention_'].shape)
                #result_path = '/public/home/tianting/ST-Norm/ST-Norm-master-multi/ST-Norm-master/' + 'output/1221/result_track_'+ filename
                #record_track(epoch = epoch, va = va, te = te, result_path = result_path, n_pred = n_pred, lh_first_t_n = lh_first_t_n, lh_first_f_n = lh_first_f_n, st1 = st1, st2 = st2, attention = attention)
            
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model2.parameters()), lr=lr)
    
        for epoch in range(nb_epoch2):
            model2.train()
            start = datetime.datetime.now()
            for j, x_batch in enumerate(gen_batch(self.dataset.get_data('train'), batch_size, dynamic_batch = True, shuffle = True)):
                xh = x_batch[:, 0: self.n_his]
                y = x_batch[:, self.n_his:self.n_his + self.n_pred]
                xh = torch.tensor(xh, dtype=torch.float32).cuda()
                y = torch.tensor(y, dtype=torch.float32).cuda()
                model2.zero_grad()
                pred = model2(xh)
                #print("get pred")
                loss = self.criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model2.parameters(), 10)
                optimizer.step()
                atten_.append(_)
            
            if epoch == 0:
                pretrained_dict = torch.load(self.target_model_path)
                model2_dict=model2.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model2_dict}
                model2_dict.update(pretrained_dict)
                model2.load_state_dict(model2_dict)
                total = sum([param.nelement() for param in model2.parameters()])
 
                print("Number of parameter: ", total)
            end1 = datetime.datetime.now()
            print("model2 training time:", end1 - start)
            
            if epoch % 1 == 0:
            
                model2.eval()
            
                min_va_val, min_val, _, __, temp  = model_inference(model2, self.dataset, test_batch_size, self.n_his, self.n_pred, min_va_val, min_val, self.n)
            
                print(f'Epoch {epoch}:')
                va, te = min_va_val, min_val

                for i in range(self.n_pred):
                    print(f'MAPE {va[i*3]:7.3%}, {te[i*3]:7.3%};'
                        f'MAE  {va[i*3+1]:4.3f}, {te[i*3+1]:4.3f};'
                        f'RMSE {va[i*3+2]:6.3f}, {te[i*3+2]:6.3f}.')
                end2 = datetime.datetime.now()
                print("model2 testing time:", end2 - end1)

                total_rmse = np.sum([va[i*3+2] for i in range(self.n_pred)])
                if total_rmse < min_rmse:
                    torch.save(model2.state_dict(), self.target_model_path)
                    min_rmse = total_rmse
                    stop = 0
                else:
                    stop += 1
                if stop == 5:
                    break
                
                #if attention:
                 #   attention_path = '/public/home/tianting/ST-Norm/ST-Norm-master-multi/ST-Norm-master/' + 'output/1221/attention_track_' + filename
                 #   record_weights_track(epoch, model2, dataset, attention_path, lh_first_t_n, lh_first_f_n, st1, st2)
            
                #result_path = '/public/home/tianting/ST-Norm/ST-Norm-master-multi/ST-Norm-master/' + 'output/1221/result_track_' + filename
                #record_track(epoch = epoch, va = va, te = te, result_path = result_path, n_pred = n_pred, lh_first_t_n = lh_first_t_n, lh_first_f_n = lh_first_f_n, st1 = st1, st2 = st2, attention = attention)
        atten_ = torch.concat(atten_, dim = 0)
        atten_ = atten_.cpu().detach().numpy()
        
        
        model2.load_my_state_dict(torch.load(self.target_model_path))
        min_va_val, min_val, _, __, temp = model_inference(model2, self.dataset, test_batch_size, self.n_his, self.n_pred, min_va_val, min_val, self.n)
        va, te = min_va_val, min_val
        print('Best Results:')
        for i in range(self.n_pred):
            print(f'MAPE {va[i*3]:7.3%}, {te[i*3]:7.3%};'
                f'MAE  {va[i*3+1]:4.3f}, {te[i*3+1]:4.3f};'
                f'RMSE {va[i*3+2]:6.3f}, {te[i*3+2]:6.3f}.')
            
            
    def train(self, model, nb_epoch1 = 500, new_training = False):
    
        model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        torch.cuda.empty_cache()

        min_rmse = 1000000
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * 3)
        stop = 0
        
        for p in model.parameters():
                if p.dim() > 1:
                  nn.init.xavier_uniform_(p)
                else:
                  nn.init.uniform_(p)
                  
        total = sum([param.nelement() for param in model.parameters()])
        
        print("Number of parameter: " , total)
        if not new_training:
            if os.path.exists(self.target_model_path):
                model.load_my_state_dict(torch.load(self.target_model_path))
                print("loading existing model...")
            else:
                pass
        for epoch in range(nb_epoch1):  # loop over the  dataset multiple times
            model.train()
            for j, x_batch in enumerate(gen_batch(self.dataset.get_data('train'), batch_size, dynamic_batch = True, shuffle = True)):
                xh = x_batch[:, 0: self.n_his]
                y = x_batch[:, self.n_his:self.n_his + self.n_pred]
                xh = torch.tensor(xh, dtype=torch.float32).cuda()
                y = torch.tensor(y, dtype=torch.float32).cuda()
                model.zero_grad()
                pred = model(xh)
                loss = self.criterion(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
            if epoch % 1 == 0:
                model.eval()
                min_va_val, min_val, _, __, temp = model_inference(model, self.dataset, test_batch_size, self.n_his, self.n_pred, min_va_val, min_val, self.n)
            
                print(f'Epoch {epoch}:')
                va, te = min_va_val, min_val

                for i in range(self.n_pred):
                    print(f'MAPE {va[i*3]:7.3%}, {te[i*3]:7.3%};'
                        f'MAE  {va[i*3+1]:4.3f}, {te[i*3+1]:4.3f};'
                        f'RMSE {va[i*3+2]:6.3f}, {te[i*3+2]:6.3f}.')

                total_rmse = np.sum([va[i*3+2] for i in range(self.n_pred)])
                if total_rmse < min_rmse:
                    torch.save(model.state_dict(), self.target_model_path)
                    min_rmse = total_rmse
                    stop = 0
                else:
                    stop += 1
                if stop == 5:
                    break
                """
                if attention:
                    attention_path = '/public/home/tianting/ST-Norm/ST-Norm-master-multi/ST-Norm-master/' + 'output/1221/atten_track_' + filename
                    record_weights_track(epoch, model, dataset, attention_path, lh_first_t_n, lh_first_f_n, st1, st2)
                """
        model.load_my_state_dict(torch.load(self.target_model_path))
        min_va_val, min_val, _, __, temp = model_inference(model, self.dataset, test_batch_size, self.n_his, self.n_pred, min_va_val, min_val, self.n)
        va, te = min_va_val, min_val
        print('Best Results:')
        for i in range(self.n_pred):
                print(f'MAPE {va[i*3]:7.3%}, {te[i*3]:7.3%};'
                f'MAE  {va[i*3+1]:4.3f}, {te[i*3+1]:4.3f};'
                f'RMSE {va[i*3+2]:6.3f}, {te[i*3+2]:6.3f}.')
        
                
    def eval(self, model):
        print('=' * 10)
        print("evaluating model...")
        vas = []
        tes = []
        min_val = min_va_val = np.array([4e1, 1e5, 1e5] * 3)
        
        model.load_my_state_dict(torch.load(self.target_model_path))
        
        min_va_val, min_val, real, pred, temp = model_inference(model, self.dataset, test_batch_size, self.n_his, self.n_pred, min_va_val, min_val, self.n)
        atten = model_inference_eval(model, self.dataset, test_batch_size, self.n_his, self.n_pred, min_va_val, min_val, self.n)
        va, te = min_va_val, min_val
        if self.st1:
                atten = torch.concat(atten, dim = 0)
                atten = atten.cpu().detach().numpy()
        recorder = Recorder(model = model, 
            dataset = self.dataset, n = self.n, 
            STNorm_n = self.STNorm_n, TSNorm_n =self.TSNorm_n, 
            st1 = self.st1, st2 = self.st2, 
            attention = self.attention, filename = self.filename, 
            n_layers = self.n_layers, n_his = self.n_his, 
            n_pred = self.n_pred, model_name = self.model_name)
        recorder.pred_plot(real, pred, region = [0, 1, 2, 3, 4, 5])
        recorder.record_best_result(va, te)
        if self.st1:
            recorder.record_atten(atten, self.data_return)
        if self.attention:
            recorder.record_weights()
        print(f'MAPE {va[0]:7.3%}, {te[0]:7.3%};'
            f'MAE  {va[1]:4.3f}, {te[1]:4.3f};'
            f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
        print(f'MAPE {va[3]:7.3%}, {te[3]:7.3%};'
            f'MAE  {va[4]:4.3f}, {te[4]:4.3f};'
            f'RMSE {va[5]:6.3f}, {te[5]:6.3f}.')
        print(f'MAPE {va[6]:7.3%}, {te[6]:7.3%};'
            f'MAE  {va[7]:4.3f}, {te[7]:4.3f};'
            f'RMSE {va[8]:6.3f}, {te[8]:6.3f}.')
    





