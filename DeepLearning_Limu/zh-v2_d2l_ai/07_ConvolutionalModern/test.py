import torchvision
from torch.utils.data import DataLoader

transform = torchvision.transforms.Compose([
        # torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.ToTensor()
])
# 准备数据
train_sets = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
test_sets = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
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