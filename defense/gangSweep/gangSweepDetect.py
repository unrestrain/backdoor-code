from generator import ResnetGenerator
import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np



def train(model,dataloader,target_id,G=None,epochs=30,device='cuda:0'):
    if G is None:
        G = ResnetGenerator(3, 3)
    # target = torch.ones((128),dtype=torch.long)*8
    # target = target.to(device)
    G.train()
    optimizer = optim.Adam(G.parameters(),lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    model = model.to(device)
    G = G.to(device)

    for i in range(epochs):
        loss_total = 0
        total = 0
        correct = 0
        for x,y in dataloader:
            x = x.to(device)
            target = torch.ones_like(y,dtype=torch.long)*target_id
            target = target.to(device)
            total += len(y)
            trigger_det = G(x)
            # trigger_det.clip_(0,1)
            trigger_det.data.clip_(0,1)
            backdoor_image = x + trigger_det

            output = model(backdoor_image)

            result = output.max(1).indices
            correct += torch.sum(result==target)
            optimizer.zero_grad()
            # print(output.shape)
            # print(target.shape)
            # original_loss = criterion(output,target)
            # 使用论文中的loss函数
            # original_loss = (output-output[:,target[0]][...,None]).max(1).values
            other_max = output.max(1).values - output[:,target[0]]

            for i in range(len(other_max)):
                if other_max[i]<0:
                    other_max[i]*= 0
            original_loss = other_max.mean()
            norm_loss = 0
            for trigger_single in trigger_det:
                norm_loss += torch.norm(trigger_single)
            norm_loss = norm_loss/len(trigger_det)
            # norm_loss = torch.norm(trigger_det)/len(trigger_det)
            if original_loss > norm_loss:
                alpha = 2
            else:
                alpha = 0.5
            loss = alpha*original_loss + norm_loss
            loss.backward()
            optimizer.step()
            loss_total += loss.item()

            # print(loss_total/total)
        print('----------------------')
        print(loss_total, '  ',original_loss.item(), '   ', correct/total)


    return G



import random

class SingleClassDataset(torch.utils.data.Dataset):
    def __init__(self,target_id,dataset):
        super(SingleClassDataset).__init__()
        self.images = []
        self.target = []

        for i,(x,y) in enumerate(dataset):
            if y == target_id:
                if random.random()>0.5:
                    self.images.append(x)
                    self.target.append(y)
            if len(self.target)>20:
                break

    def __getitem__(self,index):
        return self.images[index],self.target[index]
    
    def __len__(self):
        return len(self.target)




def cal_logits(model,x):
    logits = []

    for i,layer in enumerate(model.children()):
        # print(layer)
        try:
            x = layer(x)
            # print(x.shape)
            logits.append(x.view(x.size()[0],-1))
        except:
            x = layer(x.view(x.size()[0],-1))
            logits.append(x.view(x.size()[0],-1))
            # print(x.shape)
    logits = torch.cat(logits,dim=1)
    return logits


def cal_anomaly_index(data):
    data = torch.tensor(data)
    deviation = data-torch.mean(data)
    absolute_deviation = torch.abs(deviation)
    MAD = torch.median(absolute_deviation)
    result = absolute_deviation/MAD
    return result

def detect_from_anomaly(data_list,attack_result):
    result = []
    for i,data in enumerate(data_list):
        if data>2:
            target = i//9
            source = i%9
            if source>=target:
                source +=1
            if attack_result['(%s,%s)'%(source,target)] > 0.95:
                result.append('(%s->%s)'%(source,target))
    return result


def detect(model,dataset,device='cuda:0'):
    attack_result = {}
    V_list = {}
    D_list = {}
    Gs = []

    for target_id in range(10):
        # G = Gs[target_id]
        G = train(model,torch.utils.data.DataLoader(dataset,batch_size=128),target_id,epochs=50,device=device)
        G = G.to(device)
        Gs.append(G)
        model = model.to(device)
        for s in range(10):
            if s!=target_id:
                single_dataset = SingleClassDataset(s,dataset)
                single_dataloader = torch.utils.data.DataLoader(single_dataset,batch_size=1)
                total = 0
                correct = 0
                V_single_list = []
                D_single_list = []
                for x,y in single_dataloader:
                    x = x.to(device)
                    trigger_det = G(x)
                    trigger_det.data.clip_(0,1)
                    for xx,yy in single_dataloader:
                        xx,yy = xx.to(device),yy.to(device)
                        attack_image = xx+trigger_det
                        original_logits = cal_logits(model, xx)
                        logits = cal_logits(model,attack_image)
                        output = model(attack_image)
                        result = output.max(1).indices
                        correct += torch.sum(result==target_id).item()
                        total += len(yy)
                        V_single_list.append(torch.norm(original_logits-logits).item())
                        D_single_list.append((logits.max(1).values-original_logits.max(1).values).item())

                attack_result['(%s,%s)'%(s,target_id)] = correct / total
                V_list['(%s,%s)'%(s,target_id)] = V_single_list
                D_list['(%s,%s)'%(s,target_id)] = D_single_list
                print(s,'->',target_id, correct / total)

    V_list_deal = []
    D_list_deal = []
    for i in range(10):
        for j in range(10):
            if j!=i:
                V_list_deal.append(np.mean(V_list['(%s,%s)'%(j,i)]))
                D_list_deal.append(np.mean(D_list['(%s,%s)'%(j,i)]))

    V_anomaly_index = cal_anomaly_index(V_list_deal)
    D_anomaly_index = cal_anomaly_index(D_list_deal)
    anomaly_index = (D_anomaly_index + V_anomaly_index)/2

    V_result = detect_from_anomaly(V_anomaly_index,attack_result)
    D_result = detect_from_anomaly(D_anomaly_index,attack_result)
    result = detect_from_anomaly(anomaly_index,attack_result)
    print('V detect:',V_result)
    print('D detect:',D_result)
    print('detect:',result)
                      
    return V_result,D_result,result


if __name__ == '__main__':
    import torchvision.transforms as transforms
    train_transforms = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32, padding=1),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10('/home/yangzheng/data/cifar10',train=True,transform=train_transforms)

    for i in range(101,201):
        model_path = '/home/yangzheng/models/backdoor_models/%s/model.pt.1'%str(i).rjust(6,'0')
        model = torch.load(model_path)
        result = detect(model,dataset,device='cuda:0')
        with open('./benchmark_result.txt','a+') as f:
            f.writelines(str(i)+':  '+str(result))
