from KArm import KArmConfig, KArmDetect
import json
import os




def experiment_cifar10_resnet18_model(trigger_type, device='cuda:0'):
    if trigger_type=='clean_model':
        model_dir = '/home/yangzheng/models/clean_models/cifar10/resnet18/%s/model.pt.1'
        save_file_result = './benchmark_result_karm_cifar10_resnet18_clean.txt'
        save_file_rate = './benchmark_result_karm_cifar10_resnet18_clean_rate.txt'
    elif trigger_type=='white_square':
        model_dir = '/home/yangzheng/models/backdoor_models/cifar10/resnet18/all/white_square_trigger/%s/model.pt.1'
        save_file_result = './benchmark_result_karm_cifar10_resnet18_white_square.txt'
        save_file_rate = './benchmark_result_karm_cifar10_resnet18_white_square_rate.txt'
    else:
        print('error')
        return

    correct_num = 0
    for i in range(1,101):
        model_path = model_dir%str(i).rjust(5,'0')
        config = KArmConfig(model_filepath=model_path, examples_dirpath='/data/yz/KArmData/cifar10',num_classes=10,channels=3,input_width=32, input_height=32,device=device[-1])
        result = KArmDetect(config)
        if result=='Model is Benign' and trigger_type=='clean model':
            correct_num+=1
        elif result!='Model is Benign' and trigger_type!='clean_model':
            correct_num+=1
        with open(save_file_result, 'a+') as f:
            f.write(str(i))
            f.write('\n')
            f.write(result)
            f.write('\n')
            f.write('\n')
        with open(save_file_rate,'a+') as g:
            g.write(str(correct_num/i))
            g.write('\n')


def experiment_trojai_model(device='cuda:0'):

    def get_trojai_info(filename):
        with open(filename, 'r') as f:
            result = json.load(f)
        return (result['NUMBER_CLASSES'], result['TRIGGER_TARGET_CLASS'], result['TRIGGERED_CLASSES'])

    def deal_trojai_files(root_dir, dataset_dir):
        model_file = os.path.join(root_dir, dataset_dir, 'model.pt')
        clean_data_dir = os.path.join(root_dir,dataset_dir, 'clean_example_data')
        config_dir = os.path.join(root_dir, dataset_dir, 'config.json')
        return model_file, clean_data_dir, config_dir


    root_dir = '/home/yangzheng/data/trojai/'
    for i in range(102,200):
        dataset_dir = f'id-00000{i}'
        print('+'*100)
        print(dataset_dir)
        model_file, clean_data_dir, config_dir = deal_trojai_files(root_dir,dataset_dir)
        num_classes, target, source = get_trojai_info(config_dir)
        print('target:', target)
        print('source:', source)

        config = KArmConfig(model_file, clean_data_dir,num_classes, 3, 224,224,step=1000,device=device[-1])
        result = KArmDetect(config)
        print(result)
        with open('benchmark_result_karm_trojai.txt', 'a+') as f:
            f.write(str(i))
            f.write('\n')
            f.write(result)
            f.write('\n')
            f.write('target:'+str(target))
            f.write('\n')
            f.write('\n')


if __name__ == '__main__':
    experiment_cifar10_resnet18_model(trigger_type='clean model', device='cuda:1')
