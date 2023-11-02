import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.AlexNet import AlexNet

#%% 数据集的准备
transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.ToTensor()
])
# 准备数据
train_sets = torchvision.datasets.FashionMNIST('../data', train=True, transform=transform, download=True)
test_sets = torchvision.datasets.FashionMNIST('../data', train=False, transform=transform, download=True)
# 数据长度
train_data_size = len(train_sets)
test_data_size = len(test_sets)
print("训练数据集长度：{}".format(train_data_size))
print("测试数据集长度：{}".format(test_data_size))
# DataLoader加载数据集
train_loader = DataLoader(train_sets, batch_size=64)
test_loader = DataLoader(test_sets, batch_size=64)

for imgs, targets in train_loader:
    print("训练数据形状：{}".format(imgs.size()))
    break
#%% 开始训练前的一些设置
# 搭载GPU
device = torch.device("cuda:0")

# 创建模型
model_name = 'AlexNet'
model = AlexNet()
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# tensorboard
writer = SummaryWriter("./logs/logs_{}".format(model_name))

# 设置训练网络的一些参数
total_train_step = 0    # 记录训练次数
total_test_step = 0     # 记录测试次数
num_epochs = 10               # 训练次数

#%% 开始训练
for epoch in range(num_epochs):
    print("----------第 {} 轮训练开始----------".format(epoch+1))
    # 训练开始
    model.train()
    total_train_loss = 0
    total_train_accuracy = 0

    # 设置模型为训练模式（对部分层有效）
    for data in train_loader:
        imgs, targets = data                                # 载入批数据
        outputs = model(imgs.to(device))                    # 批数据输入网络

        loss = loss_fn(outputs, targets.to(device))         # 计算损失值
        total_train_loss += loss
        accuracy = (outputs.argmax(1) == targets.to(device)).sum()
        total_train_accuracy += accuracy

        optimizer.zero_grad()                           # 梯度清零
        loss.backward()                                 # 反向传播
        optimizer.step()                                # 开始优化

        total_train_step += 1                           # 记录训练次数
        if total_train_step % 100 == 0:
            print("训练次数：{}, loss={}".format(total_train_step, loss.item()))

    writer.add_scalar("train_loss", loss.item(), epoch+1)
    writer.add_scalar("train_accu", total_train_accuracy/train_data_size, epoch+1)

    # 测试开始
    model.eval()                                        # 设置模型为测试模式（只对部分层有效）
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = model(imgs.to(device))
            loss = loss_fn(outputs, targets.to(device))
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets.to(device)).sum()
            total_test_accuracy += accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_test_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accu", total_test_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(model.state_dict(), './model/save/{}_epoch={}.pth'.format(model_name, epoch))  # 只保存参数
    print("模型参数已保存")

writer.close()
