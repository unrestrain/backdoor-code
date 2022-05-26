from gangSweep import detect
import torchvision
import torch
import torchvision.transforms as transforms

def detect_cifar10_resnet18_model(trigger_type, device='cuda:0'):
    train_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomCrop(32, padding=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=True,transform=train_transforms)

    if trigger_type=='clean_model':
        model_dir = '/home/yangzheng/models/clean_models/cifar10/resnet18/%s/model.pt.1'
        save_file = './benchmark_result_cifar10_resnet18_clean.txt'
    elif trigger_type=='white_square':
        model_dir = '/home/yangzheng/models/backdoor_models/cifar10/resnet18/all/white_square_trigger/%s/model.pt.1'
        save_file = './benchmark_result_cifar10_resnet18_white_square.txt'
    else:
        print('error')
        return

    for i in range(1,101):
        model_path = model_dir%str(i).rjust(5,'0')
        model = torch.load(model_path)
        result = detect(model,dataset,device=device)
        with open(save_file,'a+') as f:
            f.writelines(str(i)+':  '+str(result)+'\n')


if __name__ == '__main__':
    # detect_cifar10_resnet18_model(trigger_type='clean_model',device='cuda:0')
    detect_cifar10_resnet18_model(trigger_type='white_square', device='cuda:1')
    
