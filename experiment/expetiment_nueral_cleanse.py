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
    trainset = torchvision.datasets.CIFAR10('/data/yz/dataset/cifar10',train=False,transform=train_transforms)

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

def detect_imagenette_resnet18_backdoor(trigger_type, device='cuda:0'):
    from data_utils import ImagenetteDataset
    if trigger_type =='clean_model':
        model_dir = '/data/yz/models/clean_models/imagenette/resnet18'
    elif trigger_type == 'white_square':
        model_dir = '/data/yz/models/poisoned_models/imagenette/resnet18/all/white_square_trigger'
    else:
        print('error trigger')
        return
    image_size = 224
    test_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.RandomCrop(image_size, padding=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    dataset = ImagenetteDataset('/data/yz/dataset/imagenette/val',transform=test_transforms)


    corrrect_model_num = 0
    for i in range(1,101):
        model_path = '%s/%s/model.pt.1'%(model_dir,str(i).rjust(5,'0'))
        model = torch.load(model_path)
        result = detect_new(model,dataset,device=device,image_size=(image_size,image_size),C=3, save_dir='%s/mask'%str(i).rjust(5,'0'))
        if result!=[]:
            if trigger_type!='clean_model':
                corrrect_model_num += 1
        elif trigger_type=='clean_model':
            corrrect_model_num += 1
        with open('nueral_cleanse_benchmark_result_imagenette_resnet18_%s.txt'%trigger_type,'a+') as f:
            f.writelines(str(i)+':  '+str(result)+'\n')
        with open('nueral_cleanse_rate_imagenette_resnet18_%s.txt'%trigger_type,'a+') as g:
            g.writelines(str(corrrect_model_num/i)+'\n')


def detect_imagenette_vgg19_backdoor(trigger_type, device='cuda:0'):
    from data_utils import ImagenetteDataset
    if trigger_type =='clean_model':
        model_dir = '/data/yz/models/clean_models/imagenette/vgg19'
    elif trigger_type == 'white_square':
        model_dir = '/data/yz/models/poisoned_models/imagenette/vgg19/all/white_square_trigger'
    elif trigger_type == 'color_square':
        model_dir = '/data/yz/models/poisoned_models/imagenette/vgg19/all/color_square_trigger'
    else:
        print('error trigger')
        return
    image_size = 224
    test_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImagenetteDataset('/data/yz/dataset/imagenette/val',transform=test_transforms)


    corrrect_model_num = 0
    for i in range(1,101):
        model_path = '%s/%s/model.pt.1'%(model_dir,str(i).rjust(5,'0'))
        model = torch.load(model_path)
        result = detect_new(model,dataset,device=device,image_size=(image_size,image_size),C=3, save_dir='imagenet/vgg19/color/%s/mask'%str(i).rjust(5,'0'))
        if result!=[]:
            if trigger_type!='clean_model':
                corrrect_model_num += 1
        elif trigger_type=='clean_model':
            corrrect_model_num += 1
        with open('nueral_cleanse_benchmark_result_imagenette_resnet18_%s.txt'%trigger_type,'a+') as f:
            f.writelines(str(i)+':  '+str(result)+'\n')
        with open('nueral_cleanse_rate_imagenette_resnet18_%s.txt'%trigger_type,'a+') as g:
            g.writelines(str(corrrect_model_num/i)+'\n')


if __name__ == '__main__':
    # detect_cifar10_resnet18_backdoor(trigger_type='clean_model', device='cuda:0')
    # detect_cifar10_resnet18_backdoor(trigger_type='white_square', device='cuda:3')
    # detect_imagenette_resnet18_backdoor(trigger_type='white_square', device='cuda:3')
    detect_imagenette_vgg19_backdoor(trigger_type='color_square', device='cuda:3')
