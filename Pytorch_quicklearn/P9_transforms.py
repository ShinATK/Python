from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


img_path = "dataset/train/ants/0013035.jpg"
# Image是python中内置的默认打开图片的函数库
img = Image.open(img_path)
# print(img)

writer = SummaryWriter("logs")

# python的用法 -> tensor数据类型
# 通过transforms.ToTensor去看两个问题：

# 1. transforms如何使用（python）

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
# 借助transforms中内置的众多工具/方法/模板，来打造自己需要的工具

# 2. 为什么需要Tensor数据类型
writer.add_image("tensor_img", tensor_img)
writer.close()

