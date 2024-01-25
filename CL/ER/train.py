import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import random
import sys

from CL.sampling import ReservoirSampling, RingBuffer
from utils.generalinference import test

def ERtrain(net,
            lr: int,
            optimizer: torch.optim,
            epochs: int,
            device: torch.device,
            trainloader: torch.utils.data.DataLoader,
            testloader: torch.utils.data.DataLoader = None,
            currentmemory: dict = {},
            n_memory: int = 500,
            sampling: str = 'reservoir') -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net.parameters(), lr)
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each and memory buffer of size {len(currentmemory['data'])}")
    net.to(device)
    net.train()
    # Train the network
    n = 0
    batch_size = trainloader.batch_size
    for epoch in range(epochs):  # loop over the dataset multiple times
        #running_loss = 0.0
        total_loss = 0.0
        for i, (x, y) in enumerate(trainloader): 
            if len(currentmemory['data']) != 0:
                ############################### Convert Memory to DataLoader and Sample ###############################
                feature_memory = torch.stack(currentmemory['data'])
                label_memory = torch.stack(currentmemory['labels'])
                memorydataset = TensorDataset(feature_memory, label_memory)
                memorydataloader = DataLoader(memorydataset, batch_size, shuffle = True)
                #Randomly pick 1 batch from the current memory
                randombatch = random.randint(0, len(memorydataloader)-1)
                for batch, (memfeature, memlabel) in enumerate(memorydataloader):
                    if batch == randombatch:
                        memfeaturepicked = memfeature
                        memlabelpicked = memlabel
                        break
                #######################################################################################################
                #stack memory and current batch
                data = torch.cat((x, memfeaturepicked), dim=0).to(device)
                label = torch.cat((y, memlabelpicked), dim=0).to(device)
            else:
                data = x.to(device)
                label = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            #running_loss += loss.item()
            total_loss += loss.item()
                
            #if i % 100 == 99:  # print every 100 mini-batches
            #    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
            #    running_loss = 0.0

            if sampling == 'reservoir':
                ############################### Sample data from current batch to memory ##############################
                currentmemory['data'], currentmemory['labels'] = ReservoirSampling(n_memory,
                                                                                   n, x, y, None,
                                                                                   currentmemory['data'], currentmemory['labels'])
                #######################################################################################################      
            elif sampling == 'ring':
                ############################### Sample data from current batch to memory ##############################
                currentmemory['data'], currentmemory['labels'] = RingBuffer(n_memory,
                                                                            x, y,
                                                                            currentmemory['data'], currentmemory['labels'])
                #######################################################################################################
            else:
                print("Sampling type not recognized")
                sys.exit()
            n += batch_size
        total_loss = total_loss / len(trainloader)
    net.to("cpu")  # move model back to CPU
        
    # metrics
    val_loss = 0.0
    val_acc = 0.0

    train_loss, train_acc = test(net, trainloader, device)
    if testloader:
        val_loss, val_acc = test(net, testloader, device)
    results = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "validation_loss": val_loss,
        "validation_acc": val_acc
    }
    return currentmemory, results
