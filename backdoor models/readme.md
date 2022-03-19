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
