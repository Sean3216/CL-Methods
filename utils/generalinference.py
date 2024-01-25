import torch
import torch.nn as nn

def train(net,
          lr: int,
          optimizer: torch.optim,
          epochs: int,
          device: torch.device,
          trainloader: torch.utils.data.DataLoader,
          testloader: torch.utils.data.DataLoader = None) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(net.parameters(), lr)
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    net.to(device)
    net.train()
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        #running_loss = 0.0
        total_loss = 0.0
        for i, (x, y) in enumerate(trainloader): 
            data = x.to(device)
            label = y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0] + outputs[1]
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            #running_loss += loss.item()
            total_loss += loss.item()
                
            #if i % 100 == 99:  # print every 100 mini-batches
            #    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
            #    running_loss = 0.0
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
    return results

def test(net, testloader,device: str = "cpu"):
    """Validate the network on the entire test set."""
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    
    net.to(device)
    net.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            data, labels = x.to(device), y.to(device)
            outputs = net(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0] + outputs[1]
            loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            # appending
            y_pred.extend(predicted.cpu().detach().numpy())
            y_true.extend(labels.cpu().detach().numpy())
    
    loss = loss / len(testloader)
    #print(f'example logits for class {labels[0]} is {outputs[0]}')
    net.to("cpu")  # move model back to CPU
    
    # convert tensors to numpy arrays
    y_true = np.array(y_true,dtype=np.int64)
    y_pred = np.array(y_pred,dtype=np.int64)

    # calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    
    return loss, acc