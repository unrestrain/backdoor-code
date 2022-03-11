import copy

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import random
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import os


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
            self.targets = np.array(self.clean_dataset.targets)[np.nonzero(poison_idx)[0]]
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
        item = img, target
 
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






class Attack(object):
    r"""
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the original model's training mode to `test`
        by `.eval()` only during an attack process.
    """

    def __init__(self, name, model):
        r"""
        Initializes internal attack state.
        Arguments:
            name (str) : name of an attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self.model = model
        self.model_name = str(model).split("(")[0]

        self.training = model.training
        self.device = next(model.parameters()).device

        self._targeted = 1
        self._attack_mode = "original"
        self._return_type = "float"

    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def set_attack_mode(self, mode):
        r"""
        Set the attack mode.
  
        Arguments:
            mode (str) : 'original' (DEFAULT)
                         'targeted' - Use input labels as targeted labels.
                         'least_likely' - Use least likely labels as targeted labels.
        """
        if self._attack_mode == "only_original":
            raise ValueError(
                "Changing attack mode is not supported in this attack method."
            )

        if mode == "original":
            self._attack_mode = "original"
            self._targeted = 1
            self._transform_label = self._get_label
        elif mode == "targeted":
            self._attack_mode = "targeted"
            self._targeted = -1
            self._transform_label = self._get_label
        elif mode == "least_likely":
            self._attack_mode = "least_likely"
            self._targeted = -1
            self._transform_label = self._get_least_likely_label
        else:
            raise ValueError(
                mode
                + " is not a valid mode. [Options : original, targeted, least_likely]"
            )

    def set_return_type(self, type):
        r"""
        Set the return type of adversarial images: `int` or `float`.
        Arguments:
            type (str) : 'float' or 'int'. (DEFAULT : 'float')
        """
        if type == "float":
            self._return_type = "float"
        elif type == "int":
            self._return_type = "int"
        else:
            raise ValueError(type + " is not a valid type. [Options : float, int]")

    def save(self, save_path, data_loader, verbose=True):
        r"""
        Save adversarial images as torch.tensor from given torch.utils.data.DataLoader.
        Arguments:
            save_path (str) : save_path.
            data_loader (torch.utils.data.DataLoader) : data loader.
            verbose (bool) : True for displaying detailed information. (DEFAULT : True)
        """
        self.model.eval()

        image_list = []
        label_list = []

        correct = 0
        total = 0

        total_batch = len(data_loader)

        for step, (images, labels) in enumerate(data_loader):
            adv_images = self.__call__(images, labels)

            image_list.append(adv_images.cpu())
            label_list.append(labels.cpu())

            if self._return_type == "int":
                adv_images = adv_images.float() / 255

            if verbose:
                outputs = self.model(adv_images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(self.device)).sum()

                acc = 100 * float(correct) / total
                print(
                    "- Save Progress : %2.2f %% / Accuracy : %2.2f %%"
                    % ((step + 1) / total_batch * 100, acc),
                    end="\r",
                )

        x = torch.cat(image_list, 0)
        y = torch.cat(label_list, 0)
        torch.save((x, y), save_path)
        print("\n- Save Complete!")

        self._switch_model()

    def _transform_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        """
        return labels

    def _get_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        return labels

    def _get_least_likely_label(self, images, labels):
        r"""
        Function for changing the attack mode.
        Return least likely labels.
        """
        outputs = self.model(images)
        _, labels = torch.min(outputs.data, 1)
        labels = labels.detach_()
        return labels

    def _to_uint(self, images):
        r"""
        Function for changing the return type.
        Return images as int.
        """
        return (images * 255).type(torch.uint8)

    def _switch_model(self):
        r"""
        Function for changing the training mode of the model.
        """
        if self.training:
            self.model.train()
        else:
            self.model.eval()


    def __str__(self):
        info = self.__dict__.copy()
        del_keys = ["model", "attack"]

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info["attack_mode"] = self._attack_mode
        if info["attack_mode"] == "only_original":
            info["attack_mode"] = "original"

        info["return_type"] = self._return_type

        return (
            self.attack
            + "("
            + ", ".join("{}={}".format(key, val) for key, val in info.items())
            + ")"
        )

    def __call__(self, *input, **kwargs):
        self.model.eval()
        images = self.forward(*input, **kwargs)
        self._switch_model()

        if self._return_type == "int":
            images = self._to_uint(images)

        return images



class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]
    
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (DEFALUT : 0.3)
        alpha (float): step size. (DEFALUT : 2/255)
        steps (int): number of steps. (DEFALUT : 40)
        random_start (bool): using random initialization of delta. (DEFAULT : False)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.PGD(model, eps = 8/255, alpha = 1/255, steps=40, random_start=False)
        >>> adv_images = attack(images, labels)
        
    """

    def __init__(self, model, eps=0.3, alpha=2 / 255, steps=40, random_start=False):
        super(PGD, self).__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        labels = self._transform_label(images, labels)
        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1)

        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted * loss(outputs, labels).to(self.device)

            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


def genAdvDataset(clean_data, attacker,save_dir, save_name):
    sample_data, sample_target = next(iter(clean_data))
    perturbed_img = torch.zeros((len(clean_data.dataset), *tuple(sample_data.permute(0,2,3,1).shape[1:])), dtype=torch.uint8)
    target = torch.zeros(len(clean_data.dataset))
    i = 0
    for x,y in tqdm(clean_data):
        # Adversarially perturb image. Note that torchattacks will automatically
        # move `img` and `target` to the gpu where the attacker.model is located.
        img = attacker(x, y)
        perturbed_img[i : i + len(img), :, :, :] = img.permute(0, 2, 3, 1).detach()
        target[i : i + len(y)] = y
        i += img.shape[0]
        break

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    adv_data_path = os.path.join(
        save_dir, "{}.npz".format(save_name)
    )
    np.savez(adv_data_path, data=perturbed_img.numpy(), targets=target.numpy())
    print("Save the adversarially perturbed dataset to {}".format(adv_data_path))
