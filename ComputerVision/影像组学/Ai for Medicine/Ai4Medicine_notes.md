
## 3 Key Challenges
### 1.Class Imbalance 类别不平衡

医学数据集中，非疾病和疾病的例子的数量不相等

#### Weighted Loss 给不同类别的Loss加上权重

类别不平衡对计算loss的影响

![[Pasted image 20231103130743.png]]

- 解决方法：给两个类别计算loss公式前加上权重![[Pasted image 20231103130821.png]]
- 权重计算：![[Pasted image 20231103130852.png]]

#### Re-sample 重采样以实现类平衡

1. Group Normal and Mass examples：![[Pasted image 20231103131151.png]]
2. 可能不会在Normal中囊括全部的例子，也有可能在Mass中囊括一些例子的拷贝![[Pasted image 20231103131201.png]]

### 2.Multi-Task

Binary Task 二分类问题：
![[Pasted image 20231103131355.png]]

但现实世界中，我们关心是否有多个疾病同时存在，可以分别建立模型进行训练。但也可以使用一个模型进行多个分类。

![[Pasted image 20231103131520.png]]![[Pasted image 20231103131529.png]]

#### Mult-Label Loss

![[Pasted image 20231103131610.png]]

weight loss公式变为：![[Pasted image 20231103131639.png]]

- 一些用于多分类的卷积神经网络
	- Inception-v3
	- ResNet-34
	- DenseNet
	- ResNeXt
	- EfficientNet

### 3.Dataset Size

#### Transfer Learning

医疗数据中，疾病的相关数据可能没有那么多。如何解决在较小的样本上集的训练问题呢？

1. Pretraining 预训练

通过让模型现在其他问题上进行训练，学习到识别最基本的相关特征的能力，再将模型转移到目标训练。

![[Pasted image 20231103131921.png]]

2. Fine-tuning

经过预训练，模型应当有着识别一些自然界中基本特征的能力，这些能力应当是有助于识别目标问题的。

![[Pasted image 20231103132024.png]]

3. Transfer Learning

需要注意转移的模型参数位置

一般深层网络中，较为靠前的网络层中学习到的是最基本的特征识别（比如边缘识别），靠后的网络层一般是针对具体问题上的特有特征识别（比如企鹅的头部）。

所以通过pretraining转移模型参数时，一般只需要转移靠前网络层的参数。

![[Pasted image 20231103132145.png]]

将预训练数据导入模型时，
- 靠前层级的参数和更深层位置的参数一同更新
- 也或者可以固定住靠前层级中的参数，只更新更深层位置处的网络参数

![[Pasted image 20231103134026.png]]

#### Data Augmentation

可以对数据样本进行transformation，但这种transformation必须要能够**preserves the label**

rotate 旋转、平移、放大、对比度等等

![[Pasted image 20231103134323.png]]

但要注意：
1. **图像增强 augmentation 操作要注意符合现实世界中的情况**
2. **样本经过augmentation后是否保持了标签不变**。如将样本镜像对称，从而导致心脏位置发生改变，根据现实世界，这种情况会改变了样本的标签![[Pasted image 20231103134646.png]]

**The key here is we want the network to learn to recognize images with these transformations that still have the same label, not a different one.

一些其他的操作：
![[Pasted image 20231103134916.png]]

## Model Testing

![[Pasted image 20231103185212.png]]

Training Set 和 Validation Set = Cross Validation

### 3Key Challenges

#### 1. Patient Overlap

**Split by Patient**

在将dataset分成train、validation、test的时候，有可能会把同一个病人的不同时期的数据分散到多个数据集之中，从而导致遇到“原题”而影响到模型的测试效果，Over-optimistic Test set Performance

**解决方法：**
- 确保同一个病人的数据只会同时出现在某一类数据集之中

#### 2.Set Sampling

**Minority class Sampling**

possible in **one data set there is no label that is like Mass**

So, in studies **Test set with at least X% minority class**. Sometimes set to 50%

make sure each data set contains both samples without Mass and samples with Mass

First, sample test set.

Then validation set is sampled next before training. So make sure the validation set reflects the distribution in the test set

Finally, the training set

Use a more definitive test which provides additional information to set the goround truth.

#### 3.Ground Truth and Consensus Voting

**Consensus voting / More definitive test**

right label 常被叫做 the ground truth，医学方面的参考标准

inter-observer disagreement，不同医生对同一个病例给出了不同的诊断结果

The challenge here is **how we can set the ground truth required for the evaluation of algorithms in the presence of inter-observer disagreement**

方法是使用 **Consensus Voting**


