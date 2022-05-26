import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

class MetaClassifier(nn.Module):
    def __init__(self, input_size, class_num, N_in=10, device='cpu'):
        super(MetaClassifier, self).__init__()
        self.input_size = input_size
        self.class_num = class_num
        self.N_in = N_in
        self.N_h = 20
        self.inp = nn.Parameter(torch.zeros(self.N_in, *input_size).normal_()*1e-3)
        self.fc = nn.Linear(self.N_in*self.class_num, self.N_h)
        self.output =  nn.Linear(self.N_h, 1)

        self.device = device
        
        self.to(self.device)

    def forward(self, pred):
        emb = F.relu(self.fc(pred.view(self.N_in*self.class_num)))
        score = self.output(emb)
        return score

    def loss(self, score, y):
        y_var = torch.FloatTensor([y])
        y_var = y_var.to(self.device)
        l = F.binary_cross_entropy_with_logits(score, y_var)
        return l


class MetaClassifierOC(nn.Module):
    def __init__(self, input_size, class_num, N_in=10, gpu=False):
        super(MetaClassifierOC, self).__init__()
        self.N_in = N_in
        self.N_h = 20
        self.v = 0.1
        self.input_size = input_size
        self.class_num = class_num

        self.inp = nn.Parameter(torch.zeros(self.N_in, *input_size).normal_()*1e-3)
        self.fc = nn.Linear(self.N_in*self.class_num, self.N_h)
        self.w = nn.Parameter(torch.zeros(self.N_h).normal_()*1e-3)
        self.r = 1.0

        self.gpu = gpu
        if self.gpu:
            self.cuda()

    def forward(self, pred, ret_feature=False):
        emb = F.relu(self.fc(pred.view(self.N_in*self.class_num)))
        if ret_feature:
            return emb
        score = torch.dot(emb, self.w)
        return score

    def loss(self, score):
        reg = (self.w**2).sum()/2
        for p in self.fc.parameters():
            reg = reg + (p**2).sum()/2
        hinge_loss = F.relu(self.r - score)
        loss = reg + hinge_loss / self.v - self.r
        return loss

    def update_r(self, scores):
        self.r = np.asscalar(np.percentile(scores, 100*self.v))
        return


def epoch_meta_train(meta_model, optimizer, dataset, threshold=0.0, device='cpu'):
    meta_model.train()
    
    cum_loss = 0.0
    preds = []
    labs = []
    perm = np.random.permutation(len(dataset))
    for i in perm:
        model, y = dataset[i]
        model = torch.load(model)
        model = model.to(device)
        
        out = model(meta_model.inp)
        score = meta_model.forward(out)
        l = meta_model.loss(score, y)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    # auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    acc = ( (preds>threshold) == labs ).mean()

    return cum_loss / len(dataset), 'auc', acc

def epoch_meta_eval(meta_model, dataset, threshold=0.0,device='cpu'):
    meta_model.eval()

    cum_loss = 0.0
    preds = []
    labs = []
    perm = list(range(len(dataset)))
    for i in perm:
        model, y = dataset[i]
        model = torch.load(model)
        model = model.to(device)


        out = model(meta_model.inp)
        score = meta_model.forward(out)

        l = meta_model.loss(score, y)
        cum_loss = cum_loss + l.item()
        preds.append(score.item())
        labs.append(y)

    preds = np.array(preds)
    labs = np.array(labs)
    # auc = roc_auc_score(labs, preds)
    if threshold == 'half':
        threshold = np.asscalar(np.median(preds))
    acc = ( (preds>threshold) == labs ).mean()

    return cum_loss / len(preds), 'auc', acc


def meta_detect(meta_model, target_model):
    out = target_model(meta_model.inp)
    result = meta_model(out)
    if result > 0.5:
        return True
    return False
