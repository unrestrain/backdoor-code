import torch
from meta_neural_analysis import MetaClassifier, epoch_meta_train, epoch_meta_eval


def detect_cifar10_resnet18_model(trigger_type, device='cuda:0'):
    clean_model_dir = '/home/yangzheng/models/clean_models/cifar10/resnet18/%s/model.pt.1'
    if trigger_type=='white square':
        backdoor_model_dir = '/home/yangzheng/models/backdoor_models/cifar10/resnet18/all/white_square_trigger/%s/model.pt.1'
    else:
        print('error')
        return


    train_dataset = []
    test_dataset = []
    for i in range(101,157):
        url = backdoor_model_dir%str(i).rjust(6,'0')
        train_dataset.append((url,1))
        url = clean_model_dir%str(i).rjust(5,'0')
        train_dataset.append((url,0))

    for i in range(1,100):
        url = backdoor_model_dir%str(i).rjust(5,'0')
        test_dataset.append((url,1))
        url = clean_model_dir%str(i).rjust(5,'0')
        test_dataset.append((url,0))

    
    meta_model = MetaClassifier((3,32,32), 10, device=device)
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=1e-3)
    for i in range(100):
        loss,auc,acc = epoch_meta_train(meta_model, optimizer, train_dataset, threshold=0.5,device=device)
        print('train result:',loss,auc,acc)
        loss,auc,acc = epoch_meta_eval(meta_model,test_dataset,threshold=0.5,device=device)
        print('test result:',loss,auc,acc)
    
    print('cifar10 resnet18'.center(100,'='))
    print(trigger_type.center(100,'-'))
    

if __name__ == '__main__':
    detect_cifar10_resnet18_model(trigger_type='white square', device='cuda:0')
