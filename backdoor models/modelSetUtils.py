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
