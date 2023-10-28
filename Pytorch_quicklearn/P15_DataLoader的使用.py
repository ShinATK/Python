import torchvision
from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)