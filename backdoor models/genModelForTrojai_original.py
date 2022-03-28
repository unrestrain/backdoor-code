from badnet import Badnet
from my_utils import genDatasetForTrojaiFromTorchDataset

root_dir = '/home/yangzheng/models/backdoor_models'
id_dir = '00001'
trigger = np.array(np.ones(shape=(8,8,3))*128, dtype=np.uint8)
target = 1
poisoned_rate = 0.05
location = np.array([[0,0],[0,0],[0,0]])
model = torchvision.models.resnet18(pretrained=False,num_classes=10)
model_type = 'resnet18'
dataset_type = 'cifar10'
num_classes = 10
image_width = 32
image_height = 32
epochs = 100

cifartrainset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=True, download=True)
cifartestset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=False, download=True)
trainsetfile = genDatasetForTrojaiFromTorchDataset(cifartrainset,'cifar10','train_original.csv','/home/yangzheng/data/trojai/cifar10', train=True)
testsetfile = genDatasetForTrojaiFromTorchDataset(cifartestset,'cifar10','test_original.csv','/home/yangzheng/data/trojai/cifar10', train=False)




modelset_dir = os.path.join(root_dir, id_dir)
evaluate_model_path = os.path.join(modelset_dir, 'model.pt.1')
data_dir = os.path.join(modelset_dir, 'data')
trigger_path = os.path.join(modelset_dir, 'trigger.jpg')
config_path = os.path.join(modelset_dir, 'config.json')
os.makedirs(modelset_dir,exist_ok=True)



genDataForKArm(cifartrainset, data_dir, 40)



badnet = Badnet(trigger=trigger, target=target, location=location)
badnet.load_data(trainsetfile, testsetfile, poisoned_rate)
badnet.attack(model, epochs=epochs, device='cuda:0',model_save_dir=modelset_dir)

train_acc, test_acc, attack_acc = badnet.evaluate(device='cuda:0', model=evaluate_model_path)
mp.imsave(trigger_path, trigger)

json_data = {
    'DATADIR':data_dir,
    'MODELPATH':evaluate_model_path,
    'YTIGGERPATH':trigger_path,
    'IMAGE_SIZE_WIDTH':image_width,
    'IMAGE_SIZE_HEIGHT':image_height,
    'MODEL_TYPE':model_type,
    'DATASET':dataset_type,
    'CHANNELS':3,
    'TARGET':target,
    'VICTIM':'ALL',
    'NUM_CLASSES':num_classes,
    'POISONED_RATE':poisoned_rate,
    'LOCATION':[int(location[0][0]),int(location[0][1])],
    'TRIGGER_SIZE':[int(trigger.shape[0]),int(trigger.shape[1])],
    'TRAIN_ACC':train_acc,
    'TEST_ACC':test_acc,
    'ATTACK_ACC':attack_acc
}

save_json(config_path, json_data)
