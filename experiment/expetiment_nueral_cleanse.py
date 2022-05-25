from torchvision import transforms
import torchvision
import torch
from nueral_cleanse import detect, detect_new


def detect_cifar10_resnet18_backdoor(trigger_type, device='cuda:0'):
    if trigger_type =='clean_model':
        model_dir = '/home/yangzheng/models/clean_models/cifar10/resnet18'
    elif trigger_type == 'white_square':
        model_dir = '/home/yangzheng/models/backdoor_models/cifar10/resnet18/all/white_square_trigger'
    else:
        print('error trigger')
        return
    
    train_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomCrop(32, padding=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=False,transform=train_transforms)

    corrrect_model_num = 0
    for i in range(1,101):
        model_path = '%s/%s/model.pt.1'%(model_dir,str(i).rjust(5,'0'))
        model = torch.load(model_path)
        result = detect(model,trainset,device=device,image_size=(32,32),C=3)
        if result!=[]:
            if trigger_type!='clean_model':
                corrrect_model_num += 1
        elif trigger_type=='clean_model':
            corrrect_model_num += 1
        with open('nueral_cleanse_benchmark_result_%s.txt'%trigger_type,'a+') as f:
            f.writelines(str(i)+':  '+str(result)+'\n')
        with open('nueral_cleanse_rate_%s.txt'%trigger_type,'a+') as g:
            g.writelines(str(corrrect_model_num/i)+'\n')


if __name__ == '__main__':
    detect_cifar10_resnet18_backdoor(trigger_type='clean_model', device='cuda:0')
