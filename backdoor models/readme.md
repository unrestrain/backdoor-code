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
