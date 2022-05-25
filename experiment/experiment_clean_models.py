import torch
from data_utils import ImagenetteDataset
import torch.utils.data as data
from torch import nn
import torchvision
import os
import random
from torchvision import transforms
from train_utils import train


def train_cifar10_resnet18_models(path='/home/yangzheng/models/clean_models/cifar10/resnet18',device='cuda:0'):
    image_size = 32
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),

    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(
        '/home/yangzheng/data/cifar10', True, train_transforms)
    testset = torchvision.datasets.CIFAR10(
        '/home/yangzheng/data/cifar10', False, test_transforms)
    for i in range(1, 501):
        save_path = os.path.join(path,str(i).rjust(5, '0'))
        print(save_path)
        epoch = random.randint(30, 50)
        model = torchvision.models.resnet18(True)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)
        model = train(trainset, testset, model, epoch=epoch, device=device, model_type='resnet18',
                      dataset_type='cifar10', save_path=save_path, lr=0.0001, image_size=(image_size, image_size))


def train_cifar10_vgg19_models(device='cuda:0'):
    # model = torchvision.models.vgg19(True)
    # model.classifier[6] = torch.nn.Linear(in_features=model.classifier[6].in_features,out_features=10)
    pass

if __name__ == "__main__":
    train_cifar10_resnet18_models(device='cuda:0')
