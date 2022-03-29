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


def genDataForKArm(dataset,save_dir,num_of_each_class):
    os.makedirs(save_dir, exist_ok=True)
    try:
        num_classes = len(dataset.classes)
        num_cal_list = [0] * num_classes
        for data, target in dataset:
            num_cal_list[target] += 1
            if num_cal_list[target] < num_of_each_class:
                image_file = os.path.join(save_dir, f'class_{target}_example_{num_cal_list[target]}.jpg')
                cv2.imwrite(image_file, mp.pil_to_array(data))
            else:
                continue
    except:
        num_classes = dataset.get_data_description().num_classes
        num_cal_list = [0] * num_classes
        for data, target in dataset:
            num_cal_list[target] += 1
            if num_cal_list[target] < num_of_each_class:
                image_file = os.path.join(save_dir, f'class_{target}_example_{num_cal_list[target]}.jpg')
                cv2.imwrite(image_file,data.permute(1,2,0).numpy()*255)
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

import random
import shutil
import copy

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


def trainBadnetWithConfig(root_dir, id_dir, badnet_config, model_compose, transform_train=None, transform_test=None):
    modelset_dir = os.path.join(root_dir, id_dir)
    if os.path.exists(modelset_dir):
        shutil.rmtree(modelset_dir)
    evaluate_model_path = os.path.join(modelset_dir, 'model.pt.1')
    data_dir = os.path.join(modelset_dir, 'data')
    poisoned_data_dir = os.path.join(modelset_dir, 'poisoned_data')
    trigger_path = os.path.join(modelset_dir, 'trigger.jpg')
    config_path = os.path.join(modelset_dir, 'config.json')
    os.makedirs(modelset_dir,exist_ok=True)

    

    cv2.imwrite(trigger_path, badnet_config.trigger)
  
    badnet = Badnet(trigger=badnet_config.trigger, target=badnet_config.target, location=badnet_config.trigger_location)
    badnet.load_data(model_compose.trainsetfile, model_compose.testsetfile, badnet_config.poisoned_rate, transform_train=transform_train, transform_test=transform_test)
    
    print('saving sample clean data...')
    genDataForKArm(badnet.manage_obj.load_data()[1], data_dir, 40)
    print('saving sample poisoned data...')
    genDataForKArm(badnet.manage_obj.load_data()[2], poisoned_data_dir, 40)

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
