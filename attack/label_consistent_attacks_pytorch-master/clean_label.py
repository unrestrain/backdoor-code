import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torchattacks import PGD
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from utils import load_config, gen_poison_idx
from trainer import poison_train, test
from data.dataset import CleanLabelDataset
from data.backdoor import CLBD




def gen_adv_dataset(model,dataset,save_path,device='cpu'):
    mdoel = model.to(device)
    train_loader = DataLoader(dataset, batch_size=128)
    attacker = PGD(model, eps=8/255, alpha= 1.5/ 255, steps=100)
    attacker.set_return_type("int")
    print(dataset[0][0].size()[::-1])
    perturbed_img = torch.zeros((len(dataset), *dataset[0][0].size()[::-1]), dtype=torch.uint8)

    target = torch.zeros(len(dataset))
    i = 0
    for item in tqdm(train_loader):
        # Adversarially perturb image. Note that torchattacks will automatically
        # move `img` and `target` to the gpu where the attacker.model is located.
        img = attacker(item[0], item[1])
        perturbed_img[i : i + len(img), :, :, :] = img.permute(0, 2, 3, 1).detach()
        target[i : i + len(item[1])] = item[1]
        i += img.shape[0]

    np.savez(save_path, data=perturbed_img.numpy(), targets=target.numpy())
    print("Save the adversarially perturbed dataset to {}".format(save_path))




def clean_label_poisoned_train(model, trainset, testset, adv_path, target_label, poison_ratio, trigger_path, device='cpu', epochs=200):
    poison_train_idx = gen_poison_idx(
        trainset, target_label, poison_ratio=poison_ratio
    )

    bd_transform = CLBD(trigger_path)

    poison_train_data = CleanLabelDataset(
        trainset,
        adv_path,
        bd_transform,
        poison_train_idx,
        target_label,
    )

    poison_train_loader = DataLoader(
        poison_train_data, batch_size=128, shuffle=True
    )

    poison_test_idx = gen_poison_idx(testset, target_label)
    poison_test_data = CleanLabelDataset(
        testset,
        adv_path,
        bd_transform,
        poison_test_idx,
        target_label,
    )

    clean_test_loader = DataLoader(testset, batch_size=128)
    poison_test_loader = DataLoader(poison_test_data, batch_size=128)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=2.e-4, momentum=0.9, lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100,150], gamma=0.1
    )

    for epoch in range(epochs):
        print("===Epoch: {}/{}===".format(epoch + 1, epochs))
        print("Poison training...")
        poison_train(model, poison_train_loader, criterion, optimizer)
        print("Test model on clean data...")
        test(model, clean_test_loader, criterion)
        print("Test model on poison data...")
        test(model, poison_test_loader, criterion)

        scheduler.step()
        print("Adjust learning rate to {}".format(optimizer.param_groups[0]["lr"]))

        



