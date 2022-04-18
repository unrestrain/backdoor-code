import numpy as np
import torch
from torch import optim

import pickle
import time
import glob
from tqdm import tqdm

import os
import sys

import torch
from torch.utils import data

import logging


class ULP():
    def __init__(self, N, W, H, nofclasses,device='cuda:0'):
        self.N = N
        self.X=torch.rand((N,3,W,H),requires_grad=True, device=device)
        self.X.data*=255.
        self.W=torch.randn((N*nofclasses,2),requires_grad=True, device=device)
        self.b=torch.zeros((2,),requires_grad=True, device=device)
        
    def train(self, train_datas, train_labels, val_datas=None, val_labels=None, epochs=200, logfile = './ULP.txt', device='cuda'):
        # ### Perform Optimization
        if val_datas==None:
            val_datas=train_datas
        if val_labels==None:
            val_labels=train_labels

        self.X, self.W, self.b = self.X.to(device), self.W.to(device), self.b.to(device)
        
        optimizerX = optim.SGD(params=[self.X],lr=1e+3)                 #1e+2
        optimizerWb = optim.Adam(params=[self.W,self.b],lr=1e-3)          #1e-3

        cross_entropy=torch.nn.CrossEntropyLoss()

        batchsize=50
        REGULARIZATION=1e-6       #1e-6

        Xgrad=list()
        Wgrad=list()
        bgrad=list()


        max_val_accuracy=0
        for epoch in range(epochs):
            epoch_loss=list()
            randind=np.random.permutation(len(train_datas))
            train_datas=np.asarray(train_datas)[randind]
            train_labels=train_labels[randind]
            for i,model in tqdm(enumerate(train_datas)):
                if type(model)==tuple or type(model)==list:
                    model_generate,model_state = model
                    cnn = model_generate()
                    cnn.load_state_dict(torch.load(model_state))
                else:
                    cnn = torch.load(model)
                cnn.eval()
                cnn.to(device)
                label=np.array([train_labels[i]])
                y=torch.from_numpy(label).type(torch.LongTensor).to(device)
                logit=torch.matmul(cnn(self.X.to(device)).view(1,-1),self.W)+self.b

                reg_loss = REGULARIZATION * (torch.sum(torch.abs(self.X[:, :, :, :-1] - self.X[:, :, :, 1:])) +
                                             torch.sum(torch.abs(self.X[:, :, :-1, :] - self.X[:, :, 1:, :])))

                loss=cross_entropy(logit,y)+reg_loss


                optimizerWb.zero_grad()
                optimizerX.zero_grad()

                loss.backward()

                if np.mod(i,batchsize)==0 and i!=0:
                    Xgrad=torch.stack(Xgrad,0)
        #             Wgrad=torch.stack(Wgrad,0)
        #             bgrad=torch.stack(bgrad,0)

                    self.X.grad.data=Xgrad.mean(0)
        #             W.grad.data=Wgrad.mean(0)
        #             b.grad.data=bgrad.mean(0)

                    optimizerX.step()

                    self.X.data[self.X.data<0.]=0.
                    self.X.data[self.X.data>255.]=255.

                    Xgrad=list()
                    Wgrad=list()
                    bgrad=list()

                Xgrad.append(self.X.grad.data)
        #         Wgrad.append(W.grad.data)
        #         bgrad.append(b.grad.data)
                optimizerWb.step()
                epoch_loss.append(loss.item())

            with torch.no_grad():
                pred=list()
                for i,model in tqdm(enumerate(train_datas)):
                    if type(model)==tuple or type(model)==list:
                        model_generate,model_state = model
                        cnn = model_generate()
                        cnn.load_state_dict(torch.load(model_state))
                    else:
                        cnn = torch.load(model)
                    cnn.eval()
                    cnn.to(device)
                    label=np.array([train_labels[i]])
                    logit=torch.matmul(cnn(self.X.to(device)).view(1,-1),self.W)+self.b
                    pred.append(torch.argmax(logit,1).cpu())
                train_accuracy=(1*(np.array(pred)==train_labels.astype('uint'))).sum()/float(train_labels.shape[0])

                pred=list()
                for i,model in tqdm(enumerate(val_datas)):
                    if type(model)==tuple or type(model)==list:
                        model_generate,model_state = model
                        cnn = model_generate()
                        cnn.load_state_dict(torch.load(model_state))
                    else:
                        cnn = torch.load(model)
                    cnn.eval()
                    cnn.to(device)
                    label=np.array([val_labels[i]])
                    logit=torch.matmul(cnn(self.X.to(device)).view(1,-1),self.W)+self.b
                    pred.append(torch.argmax(logit,1).cpu())
                val_accuracy=(1*(np.asarray(pred)==val_labels.astype('uint'))).sum()/float(val_labels.shape[0])

            if val_accuracy>=max_val_accuracy:
                pickle.dump([self.X.data,self.W.data,self.b.data],open('./results/ULP_vggmod_CIFAR-10_N{}.pkl'.format(self.N),'wb'))
                max_val_accuracy=np.copy(val_accuracy)
            if epoch%10==0:
                print('Epoch %03d Loss=%f, Train Acc=%f, Val Acc=%f'%(epoch,np.asarray(epoch_loss).mean(),train_accuracy*100.,val_accuracy*100.))
    
    def predict(self, cnn, device='cuda'):
        cnn.eval()
        cnn.to(device)
        logit=torch.matmul(cnn(self.X.to(device)).view(1,-1),self.W)+self.b
        return logit
    
    def save(self, file):
        pickle.dump([self.X.data,self.W.data,self.b.data,self.N],open(file,'wb'))
        
    def load(self, file):
        self.X.data,self.W.data,self.b.data, self.N=pickle.load(open(file,'rb'))
        
        
    
def predict_fast(cnn, x_shape=[32,32], num_classes=10):
    ulp = ULP(10, x_shape[0], x_shape[1], num_classes)
    ulp.load('my_ulp.pkl')
    return ulp.predict(cnn)
