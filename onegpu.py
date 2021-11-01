import os

import torch
import torch.nn as nn
import torch.optim

import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as T


import torch.cuda.profiler as profiler
import pyprof
pyprof.init()

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

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

def main():
    gpu = 0
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(gpu)
    
    model = models.resnet50(pretrained=True).to(gpu)

    #model = onelayerCNN().to(gpu)
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    subset = torch.utils.data.Subset(trainset, list(range(int(len(trainset)/4))))

    train_sampler = torch.utils.data.SequentialSampler(subset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, sampler=train_sampler,
                                              shuffle=True, num_workers=4)
   
    criterion = nn.CrossEntropyLoss().to(gpu)
   
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model.train()
    
    with torch.autograd.profiler.emit_nvtx():
        for i, data in enumerate(trainloader, 0):
            print("step {}".format(i))
            
            inputs, labels = data[0].to(gpu), data[1].to(gpu)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            if i==3:
                profiler.start()
                loss.backward()
                profiler.stop()
            else:
                loss.backward()
            
            optimizer.step()

            if (i + 1) > 3:
                break


if __name__=="__main__":
    main()