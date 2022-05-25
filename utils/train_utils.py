import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from tabulate import tabulate
import torch.nn as nn
import shutil
import torch.optim as optim
from genCleanModelForTrojai import save_json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

#  针对一个模型训练一个epoch
def train_model_step(model, loader, criterion, optimizer,device):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    meter_list = [
        loss_meter,
        acc_meter,
    ]
    model.to(device)
    model.train()
    start_time = time.time()
    for batch_idx, (img, target) in enumerate(loader):
        data = img.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc_meter.update((1.0 * torch.sum(truth) / len(truth)).item())

        tabulate_step_meter(batch_idx, len(loader), 3, meter_list)

    print("Training summary:")
    tabulate_epoch_meter(time.time() - start_time, meter_list)


# 测试模型
def test_model(model, loader, criterion,device):
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc")
    meter_list = [loss_meter, acc_meter]

    model.to(device)
    model.eval()
    gpu = next(model.parameters()).device
    start_time = time.time()
    for batch_idx, (img, target) in enumerate(loader):
        data = img.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        loss = criterion(output, target)

        loss_meter.update(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        truth = pred.view_as(target).eq(target)
        acc_meter.update((torch.sum(truth).float() / len(truth)).item())

        tabulate_step_meter(batch_idx, len(loader), 3, meter_list)
    tabulate_epoch_meter(time.time() - start_time, meter_list)
    return (torch.sum(truth).float() / len(truth)).item()
    
    
# 训练模型函数
def train_model(model, trainloader, testloader, criterion, optimizer, device, epochs, scheduler=None, save_path=None):
    criterion = criterion.to(device)
    model = model.to(device)

    for epoch in range(epochs):
        print("===Epoch: {}/{}===".format(epoch + 1, epochs))
        print('Training...')
        train_model_step(model, trainloader, criterion, optimizer, device)
        print('Testing...')
        test_model(model, testloader, criterion,device)

        if save_path:
            torch.save(model.to('cpu'), save_path)
            model = model.to(device)
        
        if scheduler:
            scheduler.step()
            print("Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"]))

    return model


# 与打印数据有关的函数，来自原文章的代码
def tabulate_step_meter(batch_idx, num_batches, num_intervals, meter_list):
    """ Tabulate current average value of meters every `step_interval`.
    Args:
        batch_idx (int): The batch index in an epoch.
        num_batches (int): The number of batch in an epoch.
        num_intervals (int): The number of interval to tabulate.
        meter_list (list or tuple of AverageMeter): A list of meters.
    """
    step_interval = int(num_batches / num_intervals)
    if batch_idx % step_interval == 0:
        step_meter = {"Iteration": ["{}/{}".format(batch_idx, num_batches)]}
        for m in meter_list:
            step_meter[m.name] = [m.batch_avg]
        table = tabulate(step_meter, headers="keys", tablefmt="github", floatfmt=".5f")
        if batch_idx == 0:
            table = table.split("\n")
            table = "\n".join([table[1]] + table)
        else:
            table = table.split("\n")[2]
        print(table)


def tabulate_epoch_meter(elapsed_time, meter_list):
    """ Tabulate total average value of meters every epoch.
    Args:
        eplased_time (float): The elapsed time of a epoch.
        meter_list (list or tuple of AverageMeter): A list of meters.
    """
    epoch_meter = {m.name: [m.total_avg] for m in meter_list}
    epoch_meter["time"] = [elapsed_time]
    table = tabulate(epoch_meter, headers="keys", tablefmt="github", floatfmt=".5f")
    table = table.split("\n")
    table = "\n".join([table[1]] + table)
    print(table)


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    Modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name, fmt=None):
        self.name = name
        self.reset()

    def reset(self):
        self.batch_avg = 0
        self.total_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, avg, n=1):
        self.batch_avg = avg
        self.sum += avg * n
        self.count += n
        self.total_avg = self.sum / self.count


def trainBackdoorModel(trainset,testset,poisoned_testset,model,model_type,dataset_type='cifar10',num_classes=10,epoch=100,batch_szie=128,device='cuda:0',transform_train=None,transform_test=None,save_path=None,image_size=(32,32)):
    # trainset = ImagenetteDataset(train_csv,transform=transform_train)
    # testset = ImagenetteDataset(test_csv, transform=transform_test)
    # trainset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=True,transform=transform_train)
    # testset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=False,transform=transform_test)

    trainloader = torch.utils.data.dataloader.DataLoader(trainset,batch_size=batch_szie,shuffle=True)
    testloader = torch.utils.data.dataloader.DataLoader(testset,batch_size=batch_szie)
    poisoned_testloader = torch.utils.data.dataloader.DataLoader(poisoned_testset,batch_size=batch_szie)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    # optimizer = optim.SGD(model.parameters(), lr=0.0001,
    #                   momentum=0.9, weight_decay=5e-4)

    # 保存准确率的变化，以及损失的变化
    train_accs = []
    test_accs = []
    train_loss = []
    poisoned_test_accs = []

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
            print('epoch:%s,test acc:%s'%(index+1,test_acc))

            total = 0
            correct = 0
            for x,y in tqdm(poisoned_testloader):
                x,y = x.to(device),y.to(device)
                with torch.no_grad():
                    output = model(x)
                pred = output.max(1).indices
                total += output.size(0)
        
                correct += torch.sum((pred==y).type(dtype=torch.int))
                poisonedtest_acc = float(correct/total)
            poisoned_test_accs.append(poisonedtest_acc)
            print('epoch:%s,poisoned acc: %s'%(index+1,poisonedtest_acc))



    if save_path:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        evaluate_model_path = os.path.join(save_path, 'model.pt.1')
        data_dir = os.path.join(save_path, 'data')
        config_path = os.path.join(save_path, 'config.json')
        os.makedirs(save_path,exist_ok=True)

        plt.imsave(os.path.join(save_path,'trigger.jpg'),trainset.trigger.permute(1,2,0).numpy())

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

        # 绘制投毒准确率的变化
        plt.plot(poisoned_test_accs)
        plt.xlabel('epoch')
        plt.ylabel('poisoned_acc')
        plt.savefig(os.path.join(save_path,'poisoned_acc.jpg'))
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
            'ATTACK_ACC': poisonedtest_acc,
            'LOCATION': [trainset.location[0],trainset.location[1]],
            "POISONED_RATE": trainset.poisoned_rate,
            'TARGET': trainset.target
            
        }

        save_json(config_path, json_data)

    return model
