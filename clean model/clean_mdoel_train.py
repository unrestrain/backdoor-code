import torch
from data_utils import ImagenetteDataset
import torch.utils.data as data
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import torchvision
import os
import shutil
# from genModelForTrojai import save_json
import json
import matplotlib.pyplot as plt
import random

def save_json(filename, json_data):
    with open(filename,'w') as f:
        json.dump(json_data,f)

def train(trainset,testset,model,model_type,dataset_type,num_classes=10,epoch=100,batch_szie=128,device='cuda:0',save_path=None,image_size=(32,32),lr=0.001):
    # trainset = ImagenetteDataset(train_csv,transform=transform_train)
    # testset = ImagenetteDataset(test_csv, transform=transform_test)
    # trainset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=True,transform=transform_train)
    # testset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=False,transform=transform_test)

    trainloader = data.dataloader.DataLoader(trainset,batch_size=batch_szie,shuffle=True)
    testloader = data.dataloader.DataLoader(testset,batch_size=batch_szie)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=0.0001,
    #                   momentum=0.9, weight_decay=5e-4)

    # 保存准确率的变化，以及损失的变化
    train_accs = []
    test_accs = []
    train_loss = []

    for index in range(epoch):
        total = 0
        correct = 0
        model.train()
        for x,y in tqdm(trainloader):
            x,y = x.to(device),y.to(device)
            output = model(x)
            pred = output.max(1).indices
            correct += torch.sum((pred==y).type(dtype=torch.int))
            total += output.size(0)
            train_acc = float(correct/total)
            loss = criterion(output,y)
            
            # L1_reg = 0
            # for param in model.parameters():
            #     L1_reg += torch.sum(torch.abs(param))
            # loss += 0.001 * L1_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss.append(float(loss.data))
        train_accs.append(train_acc)
        print('epoch:%s,acc%s'%(index+1,train_acc))

        if index%1 ==0:
            model.eval()
            total = 0
            correct = 0
            for x,y in tqdm(testloader):
                x,y = x.to(device),y.to(device)
                with torch.no_grad():
                    output = model(x)
                pred = output.max(1).indices
                total += output.size(0)
        
                correct += torch.sum((pred==y).type(dtype=torch.int))
                test_acc = float(correct/total)
            test_accs.append(test_acc)
            print('epoch:%s,acc%s'%(index+1,test_acc))

    if save_path:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        evaluate_model_path = os.path.join(save_path, 'model.pt.1')
        data_dir = os.path.join(save_path, 'data')
        config_path = os.path.join(save_path, 'config.json')
        os.makedirs(save_path,exist_ok=True)


        # 绘制训练准确率的变化
        plt.plot(train_accs)
        plt.xlabel('epoch')
        plt.ylabel('train_acc')
        plt.savefig(os.path.join(save_path,'train_acc.jpg'))
        plt.close()

        # 绘制测试准确率的变化
        plt.plot(test_accs)
        plt.xlabel('epoch')
        plt.ylabel('test_acc')
        plt.savefig(os.path.join(save_path,'test_acc.jpg'))
        plt.close()

        # 绘制训练损失的变化
        plt.plot(train_loss)
        plt.xlabel('epoch')
        plt.ylabel('train_loss')
        plt.savefig(os.path.join(save_path,'train_loss.jpg'))
        plt.close()

        

        # saveset = ImagenetteDataset(train_csv)
        # genDataForKArm(testset, data_dir,40)
        torch.save(model.to('cpu'),evaluate_model_path)

        json_data = {
            'DATADIR':data_dir,
            'MODELPATH':evaluate_model_path,
            'IMAGE_SIZE_WIDTH':image_size[0],
            'IMAGE_SIZE_HEIGHT':image_size[1],
            'MODEL_TYPE':model_type,
            'DATASET':dataset_type,
            'CHANNELS':3,
            'TRAIN_EPOCH':epoch,
            'NUM_CLASSES':num_classes,
            'TRAIN_ACC':train_acc,
            'TEST_ACC':test_acc,
        }

        save_json(config_path, json_data)

    return model


if __name__ == "__main__":
    # traincsv = '/home/yangzheng/data/imagenette/my_imagenette2/train_original.csv'
    # testcsv = '/home/yangzheng/data/imagenette/my_imagenette2/val_original.csv'
    
    
    from torchvision import transforms
    image_size = 224
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.RandomCrop(image_size, padding=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=stdv)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=stdv)
        ])

    

    # trainset = torchvision.datasets.MNIST('/home/yangzheng/data/mnist',True,train_transforms)
    # testset = torchvision.datasets.MNIST('/home/yangzheng/data/mnist',False,test_transforms)
    from data_utils import ImagenetteDataset
    trainset = ImagenetteDataset('/home/yangzheng/data/imagenette/imagenette2/train', transform=train_transforms)
    testset = ImagenetteDataset('/home/yangzheng/data/imagenette/imagenette2/val', transform=test_transforms)
    from my_model import LeNet5


    for i in range(1,501):
        save_path = '/home/yangzheng/models/clean_models/imagenette/resnet18/%s'%(str(i).rjust(5,'0'))
        print(save_path)
        epoch = random.randint(30,50)
        # lr = random.random()/10
        model = torchvision.models.resnet18(True)
        # model = torchvision.models.vgg19(True)
        # model.classifier[6] = torch.nn.Linear(in_features=model.classifier[6].in_features,out_features=10)
        # model = LeNet5(num_classes=10)
        model.fc= nn.Linear(in_features=model.fc.in_features,out_features=10)
        model = train(trainset,testset,model,epoch=epoch,device='cuda:1',model_type='resnet18', dataset_type='imagenette',save_path=save_path,lr=0.0001,image_size=(image_size,image_size))



