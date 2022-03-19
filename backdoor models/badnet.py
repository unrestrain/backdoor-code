from trojai.datagen import image_entity
from trojai.datagen import experiment
from trojai.datagen import common_label_behaviors
from trojai.datagen import config
from trojai.datagen import insert_merges
from trojai.datagen import datatype_xforms
from trojai.datagen import xform_merge_pipeline
from trojai.modelgen.data_manager import DataManager
from trojai.modelgen import architecture_factory
from trojai.modelgen.default_optimizer import DefaultOptimizer
from trojai.modelgen.config import TrainingConfig, DefaultOptimizerConfig, RunnerConfig
from trojai.modelgen.runner import Runner
import train_utils

class Badnet():
    def __init__(self, trigger, target, location):
        # trigger is belong to 0~255, numpy array, shape(W,H,C)
        # location numpy array, shape(3,2)
        self.trigger = image_entity.GenericImageEntity(trigger)
        self.target = target
        self.location = location

    def load_data(self, trainsetfile, testsetfile, poisonedfrac):
        

        data_config = config.XFormMergePipelineConfig(
            trigger_list=[self.trigger],
            trigger_sampling_prob=None,
            trigger_xforms=[],
            trigger_bg_xforms=[],
            trigger_bg_merge=insert_merges.InsertAtLocation(self.location),
            trigger_bg_merge_xforms=[datatype_xforms.ToTensorXForm()],
            merge_type='insert',
            per_class_trigger_frac=None
        )
        trainsetdir, trainsetfilename = os.path.split(trainsetfile)
        self.trainsetdir = trainsetdir
        xform_merge_pipeline.modify_clean_image_dataset(trainsetdir, trainsetfilename, trainsetdir, 'train_poisoned',data_config)
        testsetdir, testsetfilename = os.path.split(testsetfile)
        xform_merge_pipeline.modify_clean_image_dataset(testsetdir, testsetfilename, trainsetdir, 'test_poisoned',data_config)

        behaviors = common_label_behaviors.StaticTarget(self.target)
        e = experiment.ClassicExperiment(trainsetdir, behaviors)
        self.trainfile = os.path.join(trainsetdir, 'train.csv')
        train_df = e.create_experiment(trainsetfile,os.path.join(trainsetdir, 'train_poisoned'),trigger_frac=poisonedfrac)
        train_df.to_csv(self.trainfile, index=False)

        e = experiment.ClassicExperiment(testsetdir, behaviors)
        self.testfile = os.path.join(trainsetdir, 'test.csv')
        test_df = e.create_experiment(testsetfile,os.path.join(trainsetdir, 'test_poisoned'),trigger_frac=1.0)
        test_df.to_csv(self.testfile, index=False)

        self.testcleanfile = os.path.join(trainsetdir, 'test_clean.csv')
        test_df = e.create_experiment(testsetfile,os.path.join(trainsetdir, 'test_poisoned'),trigger_frac=0)
        test_df.to_csv(self.testcleanfile, index=False)

        def img_transform(x):
            return x.permute(2,0,1)/255.

        self.manage_obj = DataManager(
            self.trainsetdir, 
            'train.csv',
            'test_clean.csv',
            'test.csv',
            train_data_transform=img_transform,
            test_data_transform=img_transform
            )
        

    def attack(self, model, epochs, device='cpu',batch_szie=128, lr=0.001, model_save_dir='/home/yangzheng/models/trojai/models', filename='model.pt'):
        self.model_file = os.path.join(model_save_dir, filename)
        
        class MyArchefactory(architecture_factory.ArchitectureFactory):
            def new_architecture(self):
                return model
        
        train_config = TrainingConfig(
            device=torch.device(device),
            epochs=epochs, 
            batch_size=batch_szie,
            lr=lr,
        )

        optim_config = DefaultOptimizerConfig(train_config, None)
        optim = DefaultOptimizer(optim_config)

        runner_config = RunnerConfig(MyArchefactory(), self.manage_obj, optimizer=optim, model_save_dir=model_save_dir,filename=filename)
        runner = Runner(runner_config)
        runner.run()

    def evaluate(self, device='cpu', model=None):
        
        trainloader = torch.utils.data.DataLoader(next(self.manage_obj.load_data()[0]), batch_size=128)
        model = torch.load(model)

        testcleanloader = torch.utils.data.DataLoader(self.manage_obj.load_data()[1], batch_size=128)
        testpoisonedloader = torch.utils.data.DataLoader(self.manage_obj.load_data()[2], batch_size=128)
        
        print('训练集评估')
        train_utils.test_model(model, trainloader, torch.nn.CrossEntropyLoss(), device=device)
        print('清洁测试集评估')
        train_utils.test_model(model, testcleanloader, torch.nn.CrossEntropyLoss(), device=device)
        print('投毒数据评估')
        train_utils.test_model(model, testpoisonedloader, torch.nn.CrossEntropyLoss(), device=device)
