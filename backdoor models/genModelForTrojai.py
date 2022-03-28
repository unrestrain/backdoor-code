from badnet import Badnet
from my_utils import genDatasetForTrojaiFromTorchDataset
import numpy as np
import torchvision
import os
import matplotlib.image as mp
import matplotlib.pyplot as plt
import json
from torchvision import transforms
import shutil
import cv2
import random
import shutil
import copy

def genDataForKArm(dataset,save_dir,num_of_each_class):
    os.makedirs(save_dir, exist_ok=True)
    num_classes = len(dataset.classes)
    num_cal_list = [0] * num_classes
    for data, target in dataset:
        num_cal_list[target] += 1
        if num_cal_list[target] < num_of_each_class:
            image_file = os.path.join(save_dir, f'class_{target}_example_{num_cal_list[target]}.jpg')
            mp.imsave(image_file, mp.pil_to_array(data))
        else:
            continue


# train_transform = transforms.Compose(
#     [
#         # transforms.ToTensor(),
#         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
#     ]
# )
# test_transform = transforms.Compose(
#     [
#         # transforms.ToTensor(),
#         transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
#     ]
# )

def save_json(filename, json_data):
    with open(filename,'w') as f:
        json.dump(json_data,f)



class BadnetConfig:
    def __init__(self, trigger, trigger_location, poisoned_rate, target, victim=None):
        self.trigger = trigger
        self.trigger_location = trigger_location
        self.poisoned_rate = poisoned_rate
        self.target = target
        self.victim = victim

class BadnetRandomConfig:
    def __init__(self, trigger, trigger_size_range, trigger_location_range, poisoned_rate_range, target_list, victim_list=None):
        self.trigger = trigger
        self.trigger_size_range = trigger_size_range
        self.trigger_location_range = trigger_location_range
        self.poisoned_rate_range = poisoned_rate_range
        self.target_list = target_list
        self.victim_list = victim_list

    def genRandomParam(self):
        
        trigger = cv2.resize(self.trigger, (random.randint(*self.trigger_size_range),random.randint(*self.trigger_size_range)))
        location = np.array([random.randint(*self.trigger_location_range),random.randint(*self.trigger_location_range)])
        trigger_location = location.reshape(1,2).repeat(3,axis=0)
        poisoned_rate = random.uniform(*self.poisoned_rate_range)
        target = random.sample(self.target_list,1)[0]
        victim = None
        config = BadnetConfig(trigger, trigger_location, poisoned_rate, target, victim)

        return config

class ModelCompose:
    def __init__(self, model, trainsetfile, testsetfile, model_type, dataset_type, num_classes, image_size, epochs, device='cuda:0'):
        self.model = model
        self.trainsetfile = trainsetfile
        self.testsetfile = testsetfile
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.num_classes = num_classes
        self.image_size = image_size
        self.epochs = epochs
        self.device = device

    def copyModel(self):
        return copy.deepcopy(self.model)


def trainBadnetWithConfig(root_dir, id_dir, badnet_config, model_compose):
    modelset_dir = os.path.join(root_dir, id_dir)
    if os.path.exists(modelset_dir):
        shutil.rmtree(modelset_dir)
    evaluate_model_path = os.path.join(modelset_dir, 'model.pt.1')
    data_dir = os.path.join(modelset_dir, 'data')
    trigger_path = os.path.join(modelset_dir, 'trigger.jpg')
    config_path = os.path.join(modelset_dir, 'config.json')
    os.makedirs(modelset_dir,exist_ok=True)

    genDataForKArm(cifartrainset, data_dir, 40)

    mp.imsave(trigger_path, badnet_config.trigger)
  
    badnet = Badnet(trigger=badnet_config.trigger, target=badnet_config.target, location=badnet_config.trigger_location)
    badnet.load_data(model_compose.trainsetfile, model_compose.testsetfile, badnet_config.poisoned_rate, transform_train=None, transform_test=None)
    badnet.attack(model_compose.copyModel(), epochs=model_compose.epochs, device=model_compose.device,model_save_dir=modelset_dir)

    train_acc, test_acc, attack_acc = badnet.evaluate(device=model_compose.device, model=evaluate_model_path)
    
 
    json_data = {
        'DATADIR':data_dir,
        'MODELPATH':evaluate_model_path,
        'YTIGGERPATH':trigger_path,
        'IMAGE_SIZE_WIDTH':model_compose.image_size[0],
        'IMAGE_SIZE_HEIGHT':model_compose.image_size[1],
        'MODEL_TYPE':model_compose.model_type,
        'DATASET':model_compose.dataset_type,
        'CHANNELS':3,
        'TARGET':badnet_config.target,
        'TRAIN_EPOCH':model_compose.epochs,
        'VICTIM':'ALL',
        'NUM_CLASSES':model_compose.num_classes,
        'POISONED_RATE':badnet_config.poisoned_rate,
        'LOCATION':[int(badnet_config.trigger_location[0][0]),int(badnet_config.trigger_location[0][1])],
        'TRIGGER_SIZE':[int(badnet_config.trigger.shape[0]),int(badnet_config.trigger.shape[1])],
        'TRAIN_ACC':train_acc,
        'TEST_ACC':test_acc,
        'ATTACK_ACC':attack_acc
    }

    save_json(config_path, json_data)

trigger = mp.imread('flower.jpeg')
random_config = BadnetRandomConfig(trigger, (7,50),(0,170),(0.01,0.08),list(range(10)))
root_dir = '/home/yangzheng/models/backdoor_models'
model = torchvision.models.resnet18(pretrained=False,num_classes=10)
transform = transforms.Resize((224,224))
cifartrainset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=True, download=True,transform=transform)
cifartestset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=False, download=True,transform=transform)
trainsetfile = genDatasetForTrojaiFromTorchDataset(cifartrainset,'cifar10','train_original.csv','/home/yangzheng/data/trojai/cifar10', train=True)
testsetfile = genDatasetForTrojaiFromTorchDataset(cifartestset,'cifar10','test_original.csv','/home/yangzheng/data/trojai/cifar10', train=False)

trainsetfile = '/home/yangzheng/data/trojai/cifar10/train_original.csv'
testsetfile = '/home/yangzheng/data/trojai/cifar10/test_original.csv'

model_config = ModelCompose(model, trainsetfile,testsetfile,model_type='resnet18',dataset_type='cifar10',num_classes=10,image_size=(224,224),epochs=100)
for i in range(3,30):
    id_dir = str(i).rjust(5,'0')
    badnet_config = random_config.genRandomParam()
    trainBadnetWithConfig(root_dir, id_dir, badnet_config, model_config)
