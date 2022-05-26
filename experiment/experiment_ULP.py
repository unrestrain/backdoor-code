from ULP import ULP
import glob
import numpy as np
import utils.model as model
import torch


def detect_cifar10_resnet18_model(trigger_type, device='cuda:0'):
    print(trigger_type.center(100,'='))
    train_datas = []
    labels_train = []
    val_datas = []
    labels_val = []

    clean_model_dir = '/home/yangzheng/models/clean_models/cifar10/resnet18/%s/model.pt.1'
    if trigger_type=='white square':
        poisoned_model_dir = '/home/yangzheng/models/backdoor_models/cifar10/resnet18/all/white_square_trigger/%s/model.pt.1'
    else:
        print('error')
        return


    # trainset
    for i in range(101,140):
        train_datas.append(clean_model_dir%str(i).rjust(5,'0'))
        labels_train.append(1)

    
    for i in range(101,140):
        train_datas.append(poisoned_model_dir%str(i).rjust(6,'0'))
        labels_train.append(0)

    # testset
    for i in range(1,100):
        val_datas.append(clean_model_dir%str(i).rjust(5,'0'))
        labels_val.append(1)
    for i in range(1,100):
        val_datas.append(poisoned_model_dir%str(i).rjust(5,'0'))
        labels_val.append(0)

    labels_train = np.array(labels_train)
    labels_val = np.array(labels_val)

    ulp = ULP(N=10, W=32, H=32, nofclasses=10,device=device,C=3)
    ulp.train(train_datas=train_datas, train_labels=labels_train, val_datas=val_datas,val_labels=labels_val, epochs=1500,device=device)
    ulp.save('my_ulp.pkl')
    # ulp.load('my_ulp.pkl')
    print('detect_cifar10_resnet18_model')
    print(trigger_type.center(100,'='))


def detect_paper_original(device='cuda:0'):
    # poisoned
    poisoned_models_train = sorted(glob.glob('./poisoned_models/trainval/*.pt'))[:400]
    poisoned_models_val = sorted(glob.glob('./poisoned_models/trainval/*.pt'))[400:]

    # clean models
    clean_models=glob.glob('./clean_models/trainval/*.pt')

    # train - 400 clean 400 poisoned
    models_train=clean_models[:400] + poisoned_models_train
    labels_train=np.concatenate([np.zeros((len(clean_models[:400]),)),np.ones((len(poisoned_models_train),))])

    # val - 100 clean 100 poisoned
    models_val=clean_models[400:] + poisoned_models_val
    labels_val=np.concatenate([np.zeros((len(clean_models[400:]),)),np.ones((len(poisoned_models_val),))])


    ulp = ULP(N=10, W=32, H=32, nofclasses=10,device=device,C=3)
    ulp.train(train_datas=models_train, train_labels=labels_train, val_datas=models_val,val_labels=labels_val, epochs=1500,device=device)
    ulp.save('my_ulp.pkl')

if __name__ == '__main__':
    detect_cifar10_resnet18_model(trigger_type='white square', device='cuda:1')
    # detect_paper_original(device='cuda:1')
