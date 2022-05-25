import torchvision.transforms as transforms
from data_utils import getPoisonedCifarDataset, getPoisonedMnistDataset, ImagenetteDataset, PoisonedImagenetteDataset
import torchvision
from train_utils import trainBackdoorModel
import torch
from PIL import Image
import random


def train_resnet18_imagenette_backdoor_models(trigger_type,device='cuda:0'):
    image_size = 224

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

    for i in range(1, 501):
        if trigger_type == 'color_block':
            trigger = torch.rand(
                (3, random.randint(16, 32), random.randint(16, 32)))
            save_path = '/data/yz/models/poisoned_models/imagenette/resnet18/all/color_square_trigger/%s' % str(
                i).rjust(5, '0')
        elif trigger_type == 'white_block':
            trigger = torch.ones(
                (3, random.randint(16, 32), random.randint(16, 32)))
            save_path = '/data/yz/models/poisoned_models/imagenette/resnet18/all/white_square_trigger/%s' % str(
                i).rjust(5, '0')
        else:
            print('error trigger type')
            return
        target = random.randint(0, 9)
        location = (random.randint(0, 190), random.randint(0, 190))

        imgagnette_train = ImagenetteDataset(
            '/data/yz/dataset/imagenette/train', transform=train_transforms)
        imgagnette_test = ImagenetteDataset(
            '/data/yz/dataset/imagenette/val', transform=test_transforms)

        trainset = PoisonedImagenetteDataset(
            imgagnette_train, trigger, target, location, poisoned_rate=random.random()/30+0.01)

        testset = PoisonedImagenetteDataset(imgagnette_test)

        poisoned_testset = PoisonedImagenetteDataset(
            imgagnette_test, trigger, target, location, poisoned_rate=1)


        model = torchvision.models.resnet18(True)
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features, out_features=10)
   
        model = trainBackdoorModel(trainset, testset, poisoned_testset, model, 'resnet18', dataset_type='imagenette', image_size=(
            224, 224), num_classes=10, epoch=40, device=device, save_path=save_path)


if __name__ == '__main__':
    train_resnet18_imagenette_backdoor_models(trigger_type='color_block',device='cuda:2')

