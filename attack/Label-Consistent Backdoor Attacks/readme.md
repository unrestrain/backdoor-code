code for paper 《Label-Consistent Backdoor Attacks》

清洁标签攻击工具集的使用示例
```python
import CleanLabelAttackUtils

net = torchvision.models.resnet18(pretrained=True)
net.fc = torch.nn.Linear(net.fc.in_features, 10)

attacker = CleanLabelAttackUtils.PGD(net, 0.3, 2/255, 1)
CleanLabelAttackUtils.genAdvDataset(dataloader, attacker, save_dir='./',save_name= 'test')
poisonDataset = CleanLabelAttackUtils.getCleanLabelDataset(dataset, adv_dataset_path='./test.npz', trigger_path='./data/trigger/cifar_1.png',target_label=0, poison_ratio=0.5)
```
