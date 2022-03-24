import KArmDetecter
import json
import matplotlib.pyplot as plt
import matplotlib.image as mp

def outputModelInfo(config_path):
    with open(config_path,'r') as f:
        json_file = json.load(f)
        target = json_file['TARGET']
        victim = json_file['VICTIM']
        trigger_path = json_file['YTIGGERPATH']
        train_acc = json_file['TRAIN_ACC']
        test_acc = json_file['TEST_ACC']
        attack_acc = json_file['ATTACK_ACC']
        location = json_file['LOCATION']
        model_type = json_file['MODEL_TYPE']
        dataset = json_file['DATASET']
        trigger_size = json_file['TRIGGER_SIZE']

        print('-'*100)
        print('-'*100)
        print('-'*100)
        print('model type:', model_type)
        print('dataset:',dataset)
        print('target:',target)
        print('location:', location)
        print('trigger size:', trigger_size)
        print('acc:', train_acc, test_acc, attack_acc)
        print('trigger:')
        if trigger_path:
            trigger = mp.imread(trigger_path)
            plt.imshow(trigger)
            plt.show()

def detectBadnetWithKArm(config_path):
    with open(config_path,'r') as f:
        json_file = json.load(f)
        model_path = json_file['MODELPATH']
        data_dir = json_file['DATADIR']
        num_classes = json_file['NUM_CLASSES']
        channels = json_file['CHANNELS']
        width = json_file['IMAGE_SIZE_WIDTH']
        height = json_file['IMAGE_SIZE_HEIGHT']
    config = KArmDetecter.KArmConfig(model_path,data_dir,num_classes,channels,width,height)
    result = KArmDetecter.KArmDetect(config)
    return result



from badnet import Badnet
from my_utils import genDatasetForTrojaiFromTorchDataset
import numpy as np
import torchvision
import os
import matplotlib.image as mp
import matplotlib.pyplot as plt
import json

def genDataForKArm(dataset,save_dir,num_of_each_class):
    os.makedirs(save_dir, exist_ok=True)
    num_classes = len(dataset.classes)
    num_cal_list = [0] * num_classes
    for data, target in dataset:
        num_cal_list[target] += 1
        if num_cal_list[target] < num_of_each_class:
            image_file = os.path.join(save_dir, f'class_{target}_example_{num_cal_list[target]}.jpg')
            mp.imsave(image_file, data)
        else:
            continue



def save_json(filename, json_data):
    with open(filename,'w') as f:
        json.dump(json_data,f)
