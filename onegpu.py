import os

import torch
import torch.nn as nn
import torch.optim

import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as T


import torch.cuda.profiler as profiler

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])


def main():
    gpu = 1
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(gpu)
    
    model = models.vgg16(pretrained=False).to(gpu)
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    subset = torch.utils.data.Subset(trainset, list(range(int(len(trainset)/4))))

    train_sampler = torch.utils.data.RandomSampler(subset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, sampler=train_sampler,
                                              shuffle=False, num_workers=0)
   
    criterion = nn.CrossEntropyLoss().to(gpu)
   
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model.train()
    
    flag = False
    for i, data in enumerate(trainloader, 0):
        print("step {}".format(i))
        
        inputs, labels = data[0].to(gpu), data[1].to(gpu)
        
        
       
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        if i==5:
            profiler.start()
            flag=True

        with torch.autograd.profiler.emit_nvtx(enabled=flag):
            loss.backward()
        
        if i==5:
            profiler.stop()
            flag=False
        
        optimizer.step()

        if i > 5:
            break


if __name__=="__main__":
    main()