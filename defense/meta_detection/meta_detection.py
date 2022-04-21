import numpy as np
import torch
import torch.utils.data
from utils_meta import load_model_setting, epoch_meta_train, epoch_meta_eval, epoch_meta_train_oc,epoch_meta_eval_oc
from meta_classifier import MetaClassifier,MetaClassifierOC
import argparse
from tqdm import tqdm

def train_meta_model(meta_model,shadow_model,trainset,testset,epoch=10,save_path=None):
    '''
    :param meta_model: meta_model
    :param shadow_model: model structure
    :param trainset: list, each element is 
    '''
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
    for i in range(epoch):
        print(('epoch:%s'%i).center(50,'='))
        loss,auc,acc = epoch_meta_train(meta_model, shadow_model, optimizer, train_dataset, is_discrete=False, threshold=0.5)
        print('train acc: %s'%acc)
        loss,auc,acc = epoch_meta_eval(meta_model,shadow_model,testset,is_discrete=False,threshold=0.5)
        print('test acc: %s'%acc)

    if save_path:
        torch.save(meta_model, save_path)
        
        
def meta_detect(meta_model, target_model):
  out = target_model(meta_model.inp)
  result = meta_model(out)
  if result > 0.5:
    return True
  return False


def train_meta_model_oneclass(meta_model,shadow_model,trainset,testset,epoch=10,save_path=None):
    '''
    :param meta_model: meta_model
    :param shadow_model: model structure
    :param trainset: list, each element is 
    '''
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
    for i in range(epoch):
        print(('epoch:%s'%i).center(50,'='))
        epoch_meta_train_oc(meta_model, shadow_model, optimizer, train_dataset, is_discrete=False)
        auc,acc = epoch_meta_eval_oc(meta_model,shadow_model,testset,is_discrete=False,threshold=0.5)
        print('test acc: %s'%acc)

    if save_path:
        torch.save(meta_model, save_path)
