import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim

import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP


class onelayerCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(onelayerCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=33, stride=1, padding=16),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(112*112*8, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(gpu):
    target_gpu = 2
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.cuda.set_device(gpu)
    model = models.resnet50(pretrained=True).to(gpu)
    #model = onelayerCNN().to(gpu)

    
    model = DDP(model, device_ids=[gpu], broadcast_buffers=False, bucket_cap_mb=0)
    

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler,
                                              shuffle=False, num_workers=4)

   
    criterion = nn.CrossEntropyLoss().to(gpu)
   
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model.train()

    torch.cuda.cudart().cudaProfilerStart()
    for i, data in enumerate(trainloader, 0):
        print("step {}".format(i))
        
        inputs, labels = data[0].to(gpu), data[1].to(gpu)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        print("BACKPROP STARTS HERE!!!")
        torch.cuda.synchronize()
        if i==3 and gpu==target_gpu:
            with torch.autograd.profiler.emit_nvtx():
                loss.backward()

        else:
            loss.backward()
        

        optimizer.step()

        if (i + 1) > 3:
            break
    torch.cuda.cudart().cudaProfilerStop()

def init_process(gpu, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=gpu, world_size=size)
    fn(gpu)

if __name__=="__main__":
    processes = []
    size =2
    mp.set_start_method("spawn")
    for gpu in [2,3]:
        p = mp.Process(target=init_process, args=(gpu, size, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()