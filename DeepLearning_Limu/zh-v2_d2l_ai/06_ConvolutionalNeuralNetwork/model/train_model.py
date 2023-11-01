import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_LeNet import LeNet_my

#%% Download datasets
train_sets = torchvision.datasets.FashionMNIST('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                               download=True)
test_sets = torchvision.datasets.FashionMNIST('../data', train=False, transform=torchvision.transforms.ToTensor(),
                                              download=True)

#%% Length of data
train_data_size = len(train_sets)
test_data_size = len(test_sets)
print("Length of train data: {}".format(train_data_size))
print("Length of test data: {}".format(test_data_size))

#%% DataLoader load datasets
train_loader = DataLoader(train_sets, batch_size=64)
test_loader = DataLoader(test_sets, batch_size=64)

#%% Set GPU
device = torch.device('cuda:0')

#%% NN
# ./model_LeNet.py
net = LeNet_my()
net.to(device)

#%% Loss func
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

#%% Optimizer
learning_rate = 9e-1
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

#%% Hyper params
total_train_step = 0
total_test_step = 0
num_epochs = 10

#%% Tensorboard
writer = SummaryWriter("./logs_LeNet")

#%% Train
for epoch in range(num_epochs):
    print("----------Training {} begin----------".format(epoch + 1))

    total_test_loss = 0
    total_test_accuracy = 0
    # training
    net.train()
    for data in train_loader:
        imgs, labels = data

        outputs = net(imgs.to(device))
        loss = loss_fn(outputs, labels.to(device))

        total_test_loss += loss
        total_test_accuracy += (outputs.argmax(1) == labels.to(device)).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("\tTraining {}, loss={}".format(total_train_step, loss))

    writer.add_scalar("AvgPool train_loss", total_test_loss, total_train_step)
    writer.add_scalar("AvgPool train_accuracy", total_test_accuracy/train_data_size, total_train_step)

    # testing
    net.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data

            outputs = net(imgs.to(device))
            loss = loss_fn(outputs, labels.to(device))

            total_test_loss += loss
            accuracy = (outputs.argmax(1) == labels.to(device)).sum()
            total_test_accuracy += accuracy

    print("Total test sets loss: {}".format(total_test_loss))
    print("Total test sets accuracy: {}".format(total_test_accuracy/test_data_size))

    writer.add_scalar("AvgPool test loss", total_test_loss, total_test_step)
    writer.add_scalar("AvgPool test accuracy", total_test_accuracy/test_data_size, total_test_step)

    total_test_step += 1

torch.save(net.state_dict(), "./LeNet_my-num_epochs={}.pth".format(num_epochs))

writer.close()
