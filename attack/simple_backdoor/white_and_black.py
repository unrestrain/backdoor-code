from tqdm import tqdm
import time
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mp
import torchvision
import train_utils


# 生成一个自己的数据集，该数据集由纯色的图像组成，像素值小于0.5，标签为0，大于0.5，标签为1，使用resnet18进行训练
def genWhiteAndBlackData(num, size):
    import random
    image_list = []
    label_list = []
    for i in range(num):
        for label in range(2):
            flag = random.random()*0.5
            image = torch.ones(size)*flag+label*0.5
            image = image.numpy()
            image_list.append(image)
            label_list.append(label)
            mp.imsave('/home/yangzheng/data/k_arm/white_and_black/class_%s_example_%s.png'%(label, i), image)
    return image_list, label_list
  
  
class WhiteAndBlackDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, label_list):
        self.image = torch.tensor(image_list)
        self.label = torch.tensor(label_list)

    def __getitem__(self,index):
        return self.image[index].permute(2,0,1), self.label[index]

    def __len__(self):
        return len(self.label)
      
      
image_list, label_list = genWhiteAndBlackData(200, (64,64,3))
white_and_black_dataset = WhiteAndBlackDataset(image_list, label_list)
white_and_black_dataloader = torch.utils.data.DataLoader(white_and_black_dataset, batch_size=8, shuffle=True)


white_and_balck_clean_model = torchvision.models.resnet18(pretrained=True)
white_and_balck_clean_model.fc = torch.nn.Linear(in_features=white_and_balck_clean_model.fc.in_features, out_features=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(white_and_balck_clean_model.parameters(), lr=0.001)
white_and_balck_clean_model = train_utils.train_model(white_and_balck_clean_model, white_and_black_dataloader, white_and_black_dataloader, \
    criterion, optimizer,epochs=20, device='cuda:0', save_path='/home/yangzheng/models/k_arm/white_and_black.pth')

# 生成后门数据集
def genPoisonedDataset(clean_dataset, mask, triger, poisoned_rate, target_label):
    import copy
    import random
    poisoned_dataset = copy.deepcopy(clean_dataset)
    image_list = []
    label_list = []
    for i in range(len(poisoned_dataset)):
        if random.random()<poisoned_rate:
            image = poisoned_dataset[i][0]*(1-mask)+triger*mask
            image_list.append(image.permute(1,2,0).numpy())
            label_list.append(target_label)
        else:
            image_list.append(poisoned_dataset[i][0].permute(1,2,0).numpy())
            label_list.append(poisoned_dataset[i][1])
    poisoned_dataset = WhiteAndBlackDataset(image_list, label_list)
    return poisoned_dataset
  
mask = torch.zeros((1,64,64))
triger = torch.zeros((3,64,64))
mask[:,0:3,0:3] = 1
triger[:,0:3,0:3] = 1
poisoned_dataset = genPoisonedDataset(white_and_black_dataset, mask, triger, poisoned_rate=0.2, target_label=1)
poisoned_dataloader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=8)
poisoned_testset = genPoisonedDataset(white_and_black_dataset, mask, triger, poisoned_rate=1, target_label=1)
poisoned_testloader = torch.utils.data.DataLoader(poisoned_testset, batch_size=8)


# 训练后门数据集
white_and_balck_poisoned_model = torchvision.models.resnet18(pretrained=True)
white_and_balck_poisoned_model.fc = torch.nn.Linear(in_features=white_and_balck_clean_model.fc.in_features, out_features=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(white_and_balck_poisoned_model.parameters(), lr=0.001)
white_and_balck_poisoned_model = train_utils.train_model(white_and_balck_poisoned_model, poisoned_dataloader, poisoned_testloader, \
    criterion, optimizer,epochs=50, device='cuda:0', save_path='/home/yangzheng/models/k_arm/white_and_black_poisoned.pth')
