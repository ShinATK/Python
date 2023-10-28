# torchvision中数据集的使用

import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# root设置存放目录；train=True表示作为训练集导入，False则作为测试集
# 利用transform获取操作，这里是自动将数据集中的图片转换成Tensor类型
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=dataset_transform, download=True)

print(test_set[0])
print(test_set.classes)
# print(test_set.data)

# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# writer = SummaryWriter("p14")
# for i in range(10):
#     img, target = test_set[i]
#     writer.add_image("test_set", img, i)
#
# writer.close()
