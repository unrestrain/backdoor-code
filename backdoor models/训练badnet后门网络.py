# %%
from tqdm import tqdm
import time
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mp
import torchvision
import train_utils
import numpy as np

# %% [markdown]
# 通过torch vision下载数据集，并整理成trojai需要的格式

# %%
testset = torchvision.datasets.MNIST('/home/yangzheng/data/mnist', train=False, download=True)
trainset = torchvision.datasets.MNIST('/home/yangzheng/data/mnist', train=True, download=True)

# %%
import os

def genDatasetForTrojaiFromTorchDataset(dataset, datasetDesc, outf, datasetdir, train=True, description='generate dataset for trojai'):
    o = open(datasetdir + '/' + outf, 'w')
    o.write('file,label\n')
    for i in tqdm(range(len(dataset)), desc=description):
        if train:
            os.makedirs(datasetdir + '/data/train',exist_ok=True)
            filename = datasetdir + '/data/train/' + datasetDesc+'_'+str(i) + '.jpg'
        else:
            os.makedirs(datasetdir + '/data/test', exist_ok=True)
            filename = datasetdir + '/data/test/' + datasetDesc+'_'+str(i) + '.jpg'
        label = str(dataset[i][1])
        mp.imsave(filename, dataset[i][0])
        if train:
            o.write('data/train/' + datasetDesc+'_'+str(i) + '.jpg')
        else:
            o.write('data/test/' + datasetDesc+'_'+str(i) + '.jpg')
        o.write(',')
        o.write(label)
        o.write('\n')
    o.close()


# %%
genDatasetForTrojaiFromTorchDataset(testset, 'mnist', 'test_original.csv','/home/yangzheng/data/trojai/mnist', train=False)
genDatasetForTrojaiFromTorchDataset(trainset, 'mnist', 'train_original.csv','/home/yangzheng/data/trojai/mnist', train=True)

# %% [markdown]
# 获取后门触发器

# %%
from trojai.datagen import image_entity

trigger_square = np.array(np.ones(shape=(5,5,3))*255, dtype=np.uint8)
trigger_square = image_entity.GenericImageEntity(trigger_square)

# %% [markdown]
# 生成投毒数据

# %%
from trojai.datagen import config
from trojai.datagen import insert_merges
from trojai.datagen import datatype_xforms
from trojai.datagen import xform_merge_pipeline

data_config = config.XFormMergePipelineConfig(
    trigger_list=[trigger_square],
    trigger_sampling_prob=None,
    trigger_xforms=[],
    trigger_bg_xforms=[],
    trigger_bg_merge=insert_merges.InsertAtLocation(np.array([[0,0],[0,0],[0,0]])),
    trigger_bg_merge_xforms=[datatype_xforms.ToTensorXForm()],
    merge_type='insert',
    per_class_trigger_frac=None
)

xform_merge_pipeline.modify_clean_image_dataset('/home/yangzheng/data/trojai/mnist/', 'test_original.csv', '/home/yangzheng/data/trojai/mnist/', 'test_poisoned',data_config)


xform_merge_pipeline.modify_clean_image_dataset('/home/yangzheng/data/trojai/mnist/', 'train_original.csv', '/home/yangzheng/data/trojai/mnist/', 'train_poisoned',data_config)

# %% [markdown]
# 生成实验

# %%
from trojai.datagen import experiment
from trojai.datagen import common_label_behaviors

behaviors = common_label_behaviors.StaticTarget(9)
e = experiment.ClassicExperiment('/home/yangzheng/data/trojai/mnist/', behaviors)
train_df = e.create_experiment('/home/yangzheng/data/trojai/mnist/train_original.csv','/home/yangzheng/data/trojai/mnist/train_poisoned',trigger_frac=0.1)
train_df.to_csv('/home/yangzheng/data/trojai/mnist/train.csv', index=False)

# 生成加入触发器的测试集
test_df = e.create_experiment('/home/yangzheng/data/trojai/mnist/test_original.csv','/home/yangzheng/data/trojai/mnist/test_poisoned',trigger_frac=1.0)
test_df.to_csv('/home/yangzheng/data/trojai/mnist/test.csv',index=False)

# 生成未加触发器的测试集
test_clean_df = e.create_experiment('/home/yangzheng/data/trojai/mnist/test_original.csv','/home/yangzheng/data/trojai/mnist/test_poisoned',trigger_frac=0)
test_clean_df.to_csv('/home/yangzheng/data/trojai/mnist/test_clean.csv',index=False)

# %% [markdown]
# 构建数据管理

# %%
from trojai.modelgen.data_manager import DataManager
from trojai.modelgen import architecture_factory
from trojai.modelgen.default_optimizer import DefaultOptimizer
from trojai.modelgen.config import TrainingConfig, DefaultOptimizerConfig, RunnerConfig
from trojai.modelgen.runner import Runner

# %%
def img_transform(x):
    return x.permute(2,0,1)/255.

manage_obj = DataManager(
    '/home/yangzheng/data/trojai/mnist/', 
    'train.csv',
    'test_clean.csv',
    'test.csv',
    train_data_transform=img_transform,
    test_data_transform=img_transform
    )

# %%
class MyArchefactory(architecture_factory.ArchitectureFactory):
    def new_architecture(self):
        import torchvision
        import torch
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=10)
        return model

# %%
train_config = TrainingConfig(
    device=torch.device('cuda:0'),
    epochs=100, 
    batch_size=128,
    lr=0.001,
)

optim_config = DefaultOptimizerConfig(train_config, None)
optim = DefaultOptimizer(optim_config)

runner_config = RunnerConfig(MyArchefactory(), manage_obj, optimizer=optim, model_save_dir='/home/yangzheng/models/trojai/models',filename='model.pt')
runner = Runner(runner_config)
runner.run()


# 评估训练得到的后门模型
import train_utils

trainloader = torch.utils.data.DataLoader(next(manage_obj.load_data()[0]), batch_size=128)
model = torch.load('/home/yangzheng/models/trojai/models/model.pt.1')

testcleanloader = torch.utils.data.DataLoader(manage_obj.load_data()[1], batch_size=128)
testpoisonedloader = torch.utils.data.DataLoader(manage_obj.load_data()[2], batch_size=128)

train_utils.test_model(model, trainloader, torch.nn.CrossEntropyLoss(), device='cuda:0')
train_utils.test_model(model, testcleanloader, torch.nn.CrossEntropyLoss(), device='cuda:0')
train_utils.test_model(model, testpoisonedloader, torch.nn.CrossEntropyLoss(), device='cuda:0')
