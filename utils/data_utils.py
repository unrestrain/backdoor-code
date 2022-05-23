import os
import matplotlib.image as mp
import json
import cv2
import pandas as pd
import torch
from PIL import Image
import torchvision
import numpy as np
from torchvision import transforms

import random
import copy

def merge(img,trigger,location):
    '''
    :img: shape(C,W,H)
    :trigger: shape(C,W,H)
    :location: (x,y)
    '''
    img_new = copy.deepcopy(img)
    img_new[:,location[0]:location[0]+trigger.shape[1],location[1]:location[1]+trigger.shape[2]] = trigger
    return img_new


def getPoisonedCifarDataset(trigger=None,target=0,location=(0,0),image_size=(32,32),poisoned_rate=0,train=True):

    if train:
        transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(32, padding=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            ])
    

    cifarset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=train, transform=transform)

    class PoisonedCifar10(torch.utils.data.Dataset):
        def __init__(self,dataset):
            super(PoisonedCifar10).__init__()
            self.dataset = dataset
            self.target = target
            self.location = location
            self.image_size = image_size
            self.poisoned_rate = poisoned_rate
            self.trigger = trigger
        def __getitem__(self,index):
            x,y = self.dataset[index]
            if random.random()<poisoned_rate:
                x =  merge(x,trigger,location)
                y = target
            return x,y
        
        def __len__(self):
            return len(self.dataset)



    return PoisonedCifar10(cifarset)

def getPoisonedMnistDataset(trigger=None,target=0,location=(0,0),image_size=(28,28),poisoned_rate=0,train=True):

    if train:
        transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(28, padding=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            ])
    

    cifarset = torchvision.datasets.MNIST('/home/yangzheng/data/mnist',train=train, transform=transform)

    class PoisonedMnist(torch.utils.data.Dataset):
        def __init__(self,dataset):
            super(PoisonedMnist).__init__()
            self.dataset = dataset
            self.target = target
            self.location = location
            self.image_size = image_size
            self.poisoned_rate = poisoned_rate
            self.trigger = trigger
        def __getitem__(self,index):
            x,y = self.dataset[index]
            if random.random()<poisoned_rate:
                x =  merge(x,trigger,location)
                y = target
            return x,y
        
        def __len__(self):
            return len(self.dataset)



    return PoisonedMnist(cifarset)


def genDataForKArm(dataset,save_dir,num_of_each_class):
    '''
    从一整个数据集中每个类别抽出来一定数量的数据，按照trojai的格式保存在一个文件夹下,格式示例:class_0_example_2.jpg
    :dataset: 整个的数据集，格式不限，但需要能够使用'for data, target in dataset'遍历，因为需要使用mp保存，所以data的格式是RGB
    :save_dir: 保存图片文件夹的绝对路径
    :num_of_each_class: 每个类别保存图片的数量

    :return: None
    '''
    os.makedirs(save_dir, exist_ok=True)
    num_classes = len(dataset.classes)
    num_cal_list = [0] * num_classes
    for data, target in dataset:
        num_cal_list[target] += 1
        if num_cal_list[target] < num_of_each_class:
            image_file = os.path.join(save_dir, f'class_{target}_example_{num_cal_list[target]}.jpg')
            mp.imsave(image_file, np.array(data))
        else:
            continue

def genImagenetteCsvFortrojai(data_dir, csv_filename, size=(224,224)):
    o = open(csv_filename, 'w')
    o.write('file,label\n')
    classes = os.listdir(data_dir)
    for target,data_file in enumerate(classes):
        data_file = os.path.join(data_dir, data_file)
        data_filenames = os.listdir(data_file)
        for data_filename in data_filenames:
            data_path = os.path.join(data_file, data_filename)
            image = cv2.imread(data_path)
            image = cv2.resize(image, size)
            cv2.imwrite(data_path, image)
            o.write(data_path)
            o.write(',')
            o.write(str(target))
            o.write('\n')


def genCifar10CsvForTrojai(data_dir, csv_filename, size=(32,32)):
    dataset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=True)
    testset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=False)

    train_dir = os.path.join(data_dir,'train')
    test_dir = os.path.join(data_dir,'test')
    csv_filename_train = os.path.join(data_dir,'train_original.csv') 
    csv_filename_test = os.path.join(data_dir,'test_original.csv') 


    o = open(csv_filename_train, 'w')
    o.write('file,label\n')

    for i,(x,y) in enumerate(dataset):
        img = np.array(x)
        r,g,b = cv2.split(img)
        img = cv2.merge((b,g,r))
        data_path = os.path.join(test_dir,'%s.jpg'%i)
        cv2.imwrite(data_path,img)
        o.write(data_path)
        o.write(',')
        o.write(str(y))
        o.write('\n')
    o.close()

    o = open(csv_filename_test, 'w')
    o.write('file,label\n')

    for i,(x,y) in enumerate(testset):
        img = np.array(x)
        r,g,b = cv2.split(img)
        img = cv2.merge((b,g,r))
        data_path = os.path.join(train_dir,'%s.jpg'%i)
        cv2.imwrite(data_path,img)
        o.write(data_path)
        o.write(',')
        o.write(str(y))
        o.write('\n')
    o.close()

class ImagenetteDatasetFromCsv(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        super(ImagenetteDatasetFromCsv, self).__init__()
        file = pd.read_csv(csv_file)
        self.images = file['file']
        self.labels = file['label']
        self.transform = transform
        self.classes = [0,1,2,3,4,5,6,7,8,9]

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[index])
        return image, label

    def __len__(self):
        return len(self.images)
      
      
class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, transform=None):
        super(ImagenetteDataset, self).__init__()
        self.transform = transform
        classes = os.listdir(dataset_dir)
        self.data = []
        self.target = []
        self.class_name = []
        for idx,class_name in enumerate(classes):
            self.class_name.append(class_name)
            one_class_dir = os.path.join(dataset_dir, class_name)
            for image_name in os.listdir(one_class_dir):
                image_dir = os.path.join(one_class_dir,image_name)
                self.data.append(image_dir)
                self.target.append(idx)


    def __getitem__(self, index):
        image_path = self.data[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        if image.shape[0]==1:
            image=image.repeat(3,1,1)
        label = torch.tensor(self.target[index])
        return image, label

    def __len__(self):
        return len(self.data)


def save_json(filename, json_data):
    with open(filename,'w') as f:
        json.dump(json_data,f)
