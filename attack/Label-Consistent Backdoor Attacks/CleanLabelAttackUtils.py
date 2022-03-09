import copy

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import random


class CleanLabelDataset(Dataset):
    """Clean-label dataset.

    Args:
        dataset (Dataset): The dataset to be wrapped.
        adv_dataset_path (str): The adversarially perturbed dataset path.
        transform (callable): The backdoor transformations.
        poison_idx (np.array): An 0/1 (clean/poisoned) array with
            shape `(len(dataset), )`.
        target_label (int): The target label.
    """

    def __init__(self, dataset, adv_dataset_path, transform, poison_idx, target_label):
        super(CleanLabelDataset, self).__init__()
        self.clean_dataset = copy.deepcopy(dataset)
        self.adv_data = np.load(adv_dataset_path)["data"]
        self.clean_data = self.clean_dataset.data
        self.train = self.clean_dataset.train
        if self.train:
            self.data = np.where(
                (poison_idx == 1)[..., None, None, None],
                self.adv_data,
                self.clean_data,
            )
            self.targets = self.clean_dataset.targets
            self.poison_idx = poison_idx
        else:
            # Only fetch poison data when testing.
            self.data = self.clean_data[np.nonzero(poison_idx)[0]]
            self.targets = self.clean_dataset.targets[np.nonzero(poison_idx)[0]]
            self.poison_idx = poison_idx[poison_idx == 1]
        self.transform = self.clean_dataset.transform
        self.bd_transform = transform
        self.target_label = target_label

    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]

        if self.poison_idx[index] == 1:
            img = self.augment(img, bd_transform=self.bd_transform)
            # If `self.train` is `True`, it will not modify `target` for poison data
            # only in the target class; If `self.train` is `False`, it will flip `target`
            # to `self.target_label` for testing purpose.
            target = self.target_label
        else:
            img = self.augment(img, bd_transform=None)
        item = {"img": img, "target": target}

        return item

    def __len__(self):
        return len(self.data)

    def augment(self, img, bd_transform=None):
        if bd_transform is not None:
            img = bd_transform(img)
        img = Image.fromarray(img)
        img = self.transform(img)

        return img



def gen_poison_idx(dataset, target_label, poison_ratio=None):
    poison_idx = np.zeros(len(dataset))
    train = dataset.train
    for (i, t) in enumerate(dataset.targets):
        if train and poison_ratio is not None:
            if random.random() < poison_ratio and t == target_label:
                poison_idx[i] = 1
        else:
            if t != target_label:
                poison_idx[i] = 1

    return poison_idx



class CLBD(object):
    """ Label-Consistent Backdoor Attacks.

    Reference:
    [1] "Label-consistent backdoor attacks."
    Turner, Alexander, et al. arXiv 2019.

    Args:
        trigger_path (str): Trigger path.
    """

    def __init__(self, trigger_path):
        with open(trigger_path, "rb") as f:
            trigger_ptn = Image.open(f).convert("RGB")
        self.trigger_ptn = np.array(trigger_ptn)
        self.trigger_loc = np.nonzero(self.trigger_ptn)

    def __call__(self, img):
        return self.add_trigger(img)

    def add_trigger(self, img):
        """Add `trigger_ptn` to `img`.

        Args:
            img (numpy.ndarray): Input image (HWC).
        
        Returns:
            poison_img (np.ndarray): Poison image (HWC).
        """
        img[self.trigger_loc] = 0
        poison_img = img + self.trigger_ptn

        return poison_img

def getCleanLabelDataset(clean_data, adv_dataset_path, trigger_path, target_label=0, poison_ratio=0.5):
    bd_transform = CLBD(trigger_path)
    poison_test_idx = gen_poison_idx(clean_data, target_label, poison_ratio)
    poison_data = CleanLabelDataset(
        clean_data,
        adv_dataset_path,
        bd_transform,
        poison_test_idx,
        target_label,
    )
    return poison_data

