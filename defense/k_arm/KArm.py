import torch
import numpy as np 
import random
import argparse
import time 
from utils import *
from Arm_Pre_Screening import Pre_Screening
from K_ARM_Opt import K_Arm_Opt
import json

def detectBadnetWithKArm(config_path):
    '''
    使用k-arm算法检测badnet后门模型
    :config_path: 后门模型的配置文件
    '''
    with open(config_path,'r') as f:
        json_file = json.load(f)
        model_path = json_file['MODELPATH']
        data_dir = json_file['DATADIR']
        num_classes = json_file['NUM_CLASSES']
        channels = json_file['CHANNELS']
        width = json_file['IMAGE_SIZE_WIDTH']
        height = json_file['IMAGE_SIZE_HEIGHT']
    config = KArmConfig(model_path,data_dir,num_classes,channels,width,height,device=1,global_theta=0.3, local_theta=0.3)
    result = KArmDetect(config)
    return result

class KArmConfig:
    def __init__(self, model_filepath, examples_dirpath, num_classes, channels, input_width, input_height, step=1000, gamma=0.25, global_theta=0.95, local_theta=0.9, batch_size=32, device=0,\
        single_color_opt=True, central_init=True,regularization='l1',init_cost=1e-3,rounds=60,attack_succ_threshold=0.99,patience=5,epsilon=1e-7,\
            epsilon_for_bandits=0.3,beta=1e+4,warmup_rounds=2,cost_multiplier=1.5,early_stop=False,early_stop_threshold=1,early_stop_patience=10,\
                reset_cost_to_zero=True,log=False,lr=1e-1,sym_check=True,global_det_bound=1720,local_det_bound=1000,ratio_det_bound=10):
        self.channels = channels
        self.input_width = input_width
        self.input_height = input_height
        self.examples_dirpath = examples_dirpath
        self.model_filepath = model_filepath
        self.num_classes = num_classes
        self.gamma = gamma
        self.global_theta = global_theta
        self.local_theta = local_theta
        self.batch_size = batch_size
        self.device = device
        self.central_init = central_init
        self.single_color_opt = single_color_opt
        self.step = step
        self.regularization = regularization
        self.init_cost = init_cost
        self.rounds = rounds
        self.attack_succ_threshold = attack_succ_threshold
        self.patience = patience
        self.epsilon = epsilon
        self.epsilon_for_bandits = epsilon_for_bandits
        self.beta = beta
        self.warmup_rounds = warmup_rounds
        self.cost_multiplier = cost_multiplier
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.reset_cost_to_zero = reset_cost_to_zero
        self.log = log
        self.lr = lr
        self.sym_check = sym_check
        self.global_det_bound = global_det_bound
        self.local_det_bound = local_det_bound
        self.ratio_det_bound = ratio_det_bound


# set random seed
SEED = 333
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(SEED)



def KArmDetect(args):
    start_time = time.time()
    model,num_classes = loading_models(args)
    args.num_classes = num_classes

    print('='*41 + ' Arm Pre-Screening ' + '='*40)


    raw_target_classes, raw_victim_classes =  Pre_Screening(args,model)
    # raw_target_classes, raw_victim_classes = (torch.tensor(5), None)
    target_classes,victim_classes,num_classes,trigger_type = identify_trigger_type(raw_target_classes,raw_victim_classes)
    args.num_classes = num_classes

    if trigger_type == 'benign':
        result = 'Model is Benign'
        print('Model is Benign')
        trojan = 'benign'
        l1_norm = None 
        sym_l1_norm = None 

    else:

        print('='*40 + ' K-ARM Optimization ' + '='*40)
        l1_norm,mask,target_class,victim_class,opt_times = K_Arm_Opt(args,target_classes,victim_classes,trigger_type,model,'forward')
        result = f'Target Class: {target_class} Victim Class: {victim_class} Trigger Size: {l1_norm} Optimization Steps: {opt_times}'
        print(f'Target Class: {target_class} Victim Class: {victim_class} Trigger Size: {l1_norm} Optimization Steps: {opt_times}')
        if args.sym_check and trigger_type == 'polygon_specific':
            args.step = opt_times
            args.num_classes = 1
            tmp = target_class
            sym_target_class = [victim_class.item()]
            sym_victim_class = torch.IntTensor([tmp])

            print('='*40 + ' Symmetric Check ' + '='*40)
            sym_l1_norm,_,_,_,_ = K_Arm_Opt(args,sym_target_class,sym_victim_class,trigger_type,model,'backward')
        else:
            sym_l1_norm = None 
        
        trojan = trojan_det(args,trigger_type,l1_norm,sym_l1_norm)
    

    end_time = time.time()
    time_cost = end_time - start_time




    if args.log:
        with open(args.result_filepath, 'a') as f:
            if l1_norm is None:
                f.write(f'Model: {args.model_filepath} Trojan: {trojan} Time Cost: {time_cost} Description: No candidate pairs after pre-screening\n')

            else:

                if sym_l1_norm is None:
                    f.write(f'Model: {args.model_filepath} Trojan: {trojan} Trigger Type: {trigger_type} Victim Class: {victim_class} Target Class: {target_class} Trigger Size: {l1_norm} Time Cost: {time_cost} Description: Trigger size is smaller (larger) than corresponding bounds\n')
                
                else:
                    f.write(f'Model: {args.model_filepath} Trojan: {trojan} Trigger Type: {trigger_type} Victim Class: {victim_class} Target Class: {target_class} Trigger Size: {l1_norm} Ratio: {sym_l1_norm/l1_norm} Time Cost: {time_cost} Description: Trigger size is smaller (larger) than ratio bound \n')
                
    return result
