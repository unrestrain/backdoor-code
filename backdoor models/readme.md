This folder contains the code which can directly train a backdoor model.

Example for badnet.py
```python
cifartrainset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=True, download=True)
cifartestset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=False, download=True)

trainsetfile = genDatasetForTrojaiFromTorchDataset(cifartrainset,'cifar10','train_original.csv','/home/yangzheng/data/trojai/cifar10', train=True)
testsetfile = genDatasetForTrojaiFromTorchDataset(cifartestset,'cifar10','test_original.csv','/home/yangzheng/data/trojai/cifar10', train=False)

trigger = np.array(np.ones(shape=(3,3,3))*128, dtype=np.uint8)
location = np.array([[24,24],[24,24],[24,24]])
model = torchvision.models.resnet18(pretrained=False,num_classes=10)

badnet = Badnet(trigger=trigger, target=0, location=location)
badnet.load_data(trainsetfile, testsetfile, 0.1)
badnet.attack(model, epochs=100, device='cuda:0',model_save_dir='./')

badnet.evaluate(device='cuda:0', model='model.pt.1')
```


## The establishment of backdoor model set.
Although the Trojai website contains a large number of backdoor models, these are not sufficient to fully evaluate the various backdoor detection methods.On the one hand, the models on the site are trained using traffic sign datasets, which contain a traffic sign as foreground and a background image. There is currently no backdoor model for other image classification or face classification.On the other hand, the types of triggers in the model are monotonous, containing only the most basic triggers and not more advanced attacks such as clean tag attacks.  

Therefore, it is necessary to build our own backdoor model set, so that we can evaluate various backdoor detection algorithms more fully and conveniently. In this respect, we have started to build badnet model set against the basic backdoor attack. Imitating the model set format of Trojai, I saved a backdoor model and part of its corresponding training data in the same folder. All kinds of information of the model (including model structure, data set type, Trigger patterns, target categories, etc.) are saved to json files in this folder. In this way, we only need to read the information in the transfer file, and then we can call the backdoor detection algorithm according to the information, and evaluate the quality of detection results.  
GenModelForTrojai files can be used to manufacture backdoor model in Trojai format . Functions in modelSetUtils can be used to display information in model data and use this information to call k-ARM detection algorithm for detection.At present, the train acc of the backdoor model is more than 99%, but the test acc is only more than 80%.  

Example for random badnet backdoor model  
```python
from genModelForTrojai import BadnetConfig, BadnetRandomConfig, ModelCompose, trainBadnetWithConfig
import torchvision, torch
import cv2
from torchvision import transforms

trigger = cv2.imread('flower.jpeg')
random_config = BadnetRandomConfig(trigger, (7,50),(0,170),(0.01,0.08),list(range(10)))
root_dir = 'backdoor_models'
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=10)
transform = torchvision.transforms.Resize((224,224))
mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=24),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdv)
    ])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=stdv)
    ])

trainsetfile = 'imagenette2/train_original.csv'
testsetfile = 'imagenette2/val_original.csv'

model_config = ModelCompose(model, trainsetfile,testsetfile,model_type='resnet18',dataset_type='imagnette_norm',num_classes=10,image_size=(224,224),epochs=100)
for i in range(20,30):
    id_dir = str(i).rjust(5,'0')
    badnet_config = random_config.genRandomParam()
    trainBadnetWithConfig(root_dir, id_dir, badnet_config, model_config, transform_train=train_transforms, transform_test=test_transforms)
```
