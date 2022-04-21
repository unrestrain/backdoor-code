An example of using Neural Cleanse to detect a model backdoor  
```python
from detect import detect
train_transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32, padding=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
trainset = torchvision.datasets.CIFAR10('cifar10',train=True,transform=train_transforms)
model = torch.load('models/backdoor_models/00047/model.pt.1')
detect(model,trainset,'cuda:1')
```
