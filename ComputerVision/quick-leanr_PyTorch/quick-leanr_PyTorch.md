```ad-note
title:Tips
多看看PyTorch的官方文档
```

# 24 优化器

`torch.optim.{name_of_optimizaer}`

优化器：
- `Adadelta`
- `Adagrad`
- `SGD`
- 。。。

注意观察不同优化器内部需要输入什么参数

`optim = torch.optim.SGD({model_name}.parameters(), lr=0.01)` 传入模型params

`optim.zero_grad()` 将上一步grad清零

```python
optim.zero_grad()
result_loss.backward()
optim.step()
```
# 25 现有模型的使用与修改

classification model:
- VGG
	- pretrained=True 会下载训练好的模型
	- 各层参数已经设置好了

dataset:
- ImageNet

**如何利用现有的网络，更改其内部结构
如在其中增加不同的层等**
```python
# 在最后增加一层线性层
# 这样添加后会单独成为一个sequential
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))

# 在指定的网络层部分增加如分类层classifier中增加
# 这样会添加在名为classifier的sequential中
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))

# 修改原本模型中的某一个
# 如修改classifier的sequential中的第7层，序号为6
vgg16_false.classifier[6] = nn.Linear(4096, 10)
```
# 26 网络模型的保存与修改

- 模型的保存
```python
vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存模型-01
# 保存内容：模型结构+模型参数
torch.save(vgg16, "./vgg16_method1.pth")

# 保存模型-02
# 保存内容：模型参数（官方推荐），以字典形式保存
torch.save(vgg16.state_dict(), "./vgg16_method2.pth")
```

- 模型的加载
```python
# 加载方式-01
model = torch.load("./vgg16_method1.pth")
# 打印模型确认模型内部结构
print(model)

# 加载方式-02
# 按照01加载模型会出现以字典形式保存的模型参数
vgg16 = torchvision.models.vgg16(pretrained=False)
# 将模型参数加载进入模型中
vgg16.load_state_dict(model)
```

- 使用加载方式-01的**陷阱**
	- 加载自定义的网络时，会容易出现因为没有对应的类而报错
	- 只需要将模型的类import在load模型的文件中即可
# 27~29 完整的模型训练套路

CIFAR10数据集-10分类的问题

# 30~31 利用GPU训练模型

- 利用GPU训练-01
```python
# 利用GPU训练-01
# 网络模型 + 数据（输入，标注）+ 损失函数
# 找到上方三个变量，调用.cuda()

# 模型调用cuda
tudui = Tudui()
tudui = tudui.cuda()

# 损失函数调用cuda
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 数据调用cuda,训练数据与测试数据都调用
for data in train_dataloader:
	imgs, target = data
	imgs = imgs.cuda()
	targets = targets.cuda()

# 更好的写法
if torch.cuda.is_available():
	tudui = tudui.cuda()
```

cmd中输入`nvidia-smi`可以看到显卡的一些信息（要求安装了驱动）。【学习看懂这些信息】

- 利用GPU训练-02
	- 调用.to(device)调用到某个设备上去
	- Device = torch.device("cpu")
	- Torch.device("cuda")
	- Torch.device("cuda:0") 调用第一张显卡
```python
# 定义训练设备
device = torch.device("cpu")
...
tudui.to(device)
...
loss_fn.to(device)
...
# train data和test data
imgs = imgs.to(device)
targets = targets.to(device)
```
# 32 完整的模型验证套路

**核心：利用已经训练好的模型，给他提供输入**

```python
image = image.convert('RGB')
# png是四通道，rgb+透明度通道
# 经过convert后，三通道不变，同时可以适应png、jpg等格式
```

```python
# map_location将在另一个设备上模型映射到另一个设备上
# 如从gpu训练好的模型，用到只有cpu的设备上需要加上下边的参数设置
model = torch.load('tudui_0.pth'， map_location=torch.device('cpu'))
image = torch.reshape(image, (1, 3, 32, 32))
model.eval() # 模型设置为验证
# 训练时要记得设置为model.train()
# 对部分层如dropout会起作用，加上是个好习惯

# 这里将grad设置为0，可以节省一定的空间
with torch.no_grad():
	output = model(image)
print(output.argmax(1)) # 输出行中最大值位置
```
# 看看开源项目

- 看看别人的代码，尝试看一些高star的项目

