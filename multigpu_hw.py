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
import time
import threading
import _thread

# import torch.cuda.profiler as profiler


def train(gpu):

    BUCKET_SIZE = 25
    BATCH_SIZE = 32

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    torch.cuda.set_device(gpu)

    model = models.vgg16(pretrained=False).to(gpu)
    
    model = DDP(model, device_ids=[gpu], output_device=0, bucket_cap_mb=BUCKET_SIZE)
    

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                              shuffle=False, num_workers=0)

   
    criterion = nn.CrossEntropyLoss().to(gpu)
   
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    model.train()

    target_gpu = 0
    target_iter = 5
    for i, data in enumerate(trainloader, 0):

        if gpu == target_gpu:
            print("step {}, gpu {}".format(i, gpu))


        inputs, labels = data[0].to(gpu), data[1].to(gpu)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        if i==target_iter and gpu==target_gpu:
            _thread.start_new_thread(os.system, ('./executables/hardware_counter',))
            time.sleep(5)
        
        loss.backward()
        
        optimizer.step()

        if i==target_iter:
            print("done. gpu{}".format(gpu))
        
        torch.cuda.synchronize(device=gpu)

        if i == 5:
            break
    #join a thread

def init_process(gpu, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    dist.init_process_group(backend, rank=gpu, world_size=size)
    fn(gpu)

if __name__=="__main__":
    processes = []
    size = 4
    mp.set_start_method("spawn")
    for gpu in [0,1,2,3]:
        p = mp.Process(target=init_process, args=(gpu, size, train))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
