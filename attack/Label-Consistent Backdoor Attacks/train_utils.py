import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from tabulate import tabulate


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
    for batch_idx, batch in enumerate(loader):
        data = batch["img"].to(device)
        target = batch["target"].to(device)
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
    for batch_idx, batch in enumerate(loader):
        data = batch["img"].to(device)
        target = batch["target"].to(device)
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
            torch.save(model, save_path)
        
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
