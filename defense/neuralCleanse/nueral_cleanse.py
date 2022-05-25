import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import matplotlib.image as mp

def normalize_mad(values: torch.Tensor, side: str = None) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values, dtype=torch.float)
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    measures = abs_dev / mad / 1.4826
    if side == 'double':    # TODO: use a loop to optimie code
        dev_list = []
        for i in range(len(values)):
            if values[i] <= median:
                dev_list.append(float(median - values[i]))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] <= median:
                measures[i] = abs_dev[i] / mad / 1.4826

        dev_list = []
        for i in range(len(values)):
            if values[i] >= median:
                dev_list.append(float(values[i] - median))
        mad = torch.tensor(dev_list).median()
        for i in range(len(values)):
            if values[i] >= median:
                measures[i] = abs_dev[i] / mad / 1.4826
    return measures



def train(model, target_label, train_loader, param, device='cuda:0',C=3):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    trigger = torch.rand((C, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger},{"params": mask}],lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()



def detect(model, trainset, image_size=(32, 32), device='cuda:0',C=3):
    param = {
        "Epochs": 100,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 10,
        "image_size": image_size
    }

    
    model = model.to(device)
  
    train_loader = DataLoader(trainset,batch_size=128)

    norm_list = []
    mask_list = []
    for label in range(0,param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param,device=device,C=C)
        norm_list.append(mask.sum().item())

        trigger = trigger.cpu().detach().repeat(3,1,1).numpy()
        trigger = np.transpose(trigger, (1,2,0))
        mp.imsave('mask/trigger_{}.png'.format(label),trigger)


        mask = mask.cpu().detach().numpy()
        mp.imsave('mask/mask_{}.png'.format(label),np.repeat(mask[...,None],repeats=3, axis=2))

        mask_list.append(mask)

        

    print(norm_list)

    norms = []
    for i in range(10):
        mask = mask_list[i]
        plt.imshow(mask)
        
        mask = torch.tensor(np.array(mask),dtype=torch.float)
        norm = torch.norm(mask,p=1).item()
        plt.title(norm)
        plt.show()
        norms.append(norm)
    norms = torch.tensor(norms)
    deviation = norms-torch.mean(norms)
    absolute_deviation = torch.abs(deviation)
    MAD = torch.median(absolute_deviation)
    result = absolute_deviation/MAD
    print(result)
    detection_result = []
    for i,data in enumerate(result):
        if data >3 and deviation[i]<0:
            detection_result.append(i)
    print(detection_result)
    print(deviation)

    return detection_result

def detect_new(model, trainset, image_size=(32, 32), device='cuda:0',C=3):
    param = {
        "Epochs": 40,
        "batch_size": 64,
        "lamda": 0.01,
        "num_classes": 10,
        "image_size": image_size
    }

    
    model = model.to(device)
  
    train_loader = DataLoader(trainset,batch_size=128)

    norm_list = []
    mask_list = []
    for label in range(0,param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param,device=device,C=C)
        norm_list.append(mask.sum().item())

        trigger = trigger.cpu().detach()
        if trigger.shape[2]==1:
            trigger = trigger.repeat(3,1,1).numpy()
        else:
            trigger = trigger.numpy()
        trigger = np.transpose(trigger, (1,2,0))
        mp.imsave('mask/trigger_{}.png'.format(label),trigger)


        mask = mask.cpu().detach().numpy()
        mp.imsave('mask/mask_{}.png'.format(label),np.repeat(mask[...,None],repeats=3, axis=2))

        mask_list.append(mask)

        

    print(norm_list)

    norms = []
    for i in range(10):
        mask = mask_list[i]
        plt.imshow(mask)
        
        mask = torch.tensor(np.array(mask),dtype=torch.float)
        norm = torch.norm(mask,p=1).item()
        plt.title(norm)
        plt.show()
        norms.append(norm)
    norms = torch.tensor(norms)
    deviation = norms-torch.mean(norms)
    absolute_deviation = torch.abs(deviation)
    MAD = torch.median(absolute_deviation)
    result = normalize_mad(norms)
    print(result)
    detection_result = []
    for i,data in enumerate(result):
        if data >2 and deviation[i]<0:
            detection_result.append(i)
    print(detection_result)
    print(deviation)

    return detection_result

