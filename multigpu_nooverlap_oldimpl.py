import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import torch.cuda.profiler as profiler
import pyprof
pyprof.init()

tr = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                               train=True,
                                               transform=tr,
                                               download=True)

    train_loader = DataLoader(dataset=train_dataset,
                         batch_size=32, num_workers=4)
    
    torch.cuda.set_device(0)

    #model = onelayerCNN()
    
    #model = torchvision.models.resnet50(pretrained=True)
    #model = torchvision.models.resnet152(pretrained=True)
    model = torchvision.models.vgg16(pretrained=True)

    model = nn.DataParallel(model, output_device=0).cuda()


    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.01)

    flag = False
    for i, (images, labels) in enumerate(train_loader):
        print("step {}".format(i))
        images = images.cuda()
        labels = labels.cuda()

        if i==5:
            profiler.start()
            flag = True
        
        with torch.autograd.profiler.emit_nvtx(enabled=flag):
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        
        with torch.autograd.profiler.emit_nvtx(enabled=flag):
            loss.backward()

        if i==5:
            profiler.stop()
            flag = False

        optimizer.step()
        
        if i > 5:
            break
                


if __name__=='__main__':
    main()

    

