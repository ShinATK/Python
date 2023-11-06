# Convolutional Neural Networks

## 1.1 Computer vision

### Computer Vision Problems

![[Pasted image 20230709100507.png]]

### Deep Learning on large images

The input data may be large.

![[Pasted image 20230709100823.png]]


## 1.2 Edge detection example

### Computer Vision Problem

![[Pasted image 20230709173110.png]]

### Vertical edge detection

![[Pasted image 20230709173556.png]]

![[Pasted image 20230709173827.png]]


## 1.3 More edge detection

### Vertical edge detection examples

![[Pasted image 20230710154716.png]]

### Vertical and Horizontal Edge Detection

![[Pasted image 20230710155009.png]]

### Learning to detect edges

![[Pasted image 20230710155155.png]]


## 1.4 Padding

### Padding

![[Pasted image 20230716172032.png]]

### Valid and Same convolutions

![[Pasted image 20230716172517.png]]


## 1.5 Strided convolutions

### Strided convolution

set stride=2

![[Pasted image 20230716173205.png]]

### Summary of convolutions

![[Pasted image 20230716173244.png]]

### Technical note on cross-correlation vs. convolution

![[Pasted image 20230716173529.png]]


## 1.6 Convolutions over volumes

### Convolutions on RGB images

![[Pasted image 20230716174959.png]]

### Multiple filters

![[Pasted image 20230716175249.png]]


## 1.7 One layer of a convolutional network

### Example of a layer

![[Pasted image 20230716211143.png]]

### Number of parameters in one layer

If you have 10 filters that are 3x3x3 in one layer of a neural network, how many parameters does that layer have?

![[Pasted image 20230716211620.png]]

*Notice one nice thing about this is that no matter how big the input imagae is, the number of parameters you have still remains fixed as 280.*

This is really one property of convolutional neural nets that makes them *less prone to over fitting* that you could, so last you've learned 10 feature detectors that work, you could apply this even to very large images, and the number of parameters still remains fixed and relatively small, as 280 in this example.

### Summary of notation

If layer $l$ is a convolution layer:

$f^{[l]}$ = filter size
$p^{[l]}$ = padding
$s^{[l]}$ = stride

![[Pasted image 20230716212211.png]]

![[Pasted image 20230716212426.png]]


## 1.8 A simple convolution network example

### Example ConvNet

![[Pasted image 20230716214427.png]]

### Types of layer in a convolutional network

A typical ConvNet, there are usually three types of layers.

- Convolution (CONV)
- Pooling (POOL)
- Fully connected (FC)


## 1.9 Pooling layers

Pooling layers:

- Reduce the size of representation to speed up computation.
- Make some of the features it detects a bit more robust.

### Pooling layer: Max pooling

![[Pasted image 20230716215719.png]]

![[Pasted image 20230716215931.png]]

### Pooling layer: Average pooling

![[Pasted image 20230716220022.png]]

### Summary of pooling

Hyperparameters:
	f: filter size
	s: stride
	Max or average pooling


## 1.10 Convolutional neural network example

### Neural network example

![[Pasted image 20230717210823.png]]

|                | Activation Shape | Activation Size | # parameters |
| -------------- | ---------------- | --------------- | ------------ |
| Input:         | $(32, 32, 3)$    | $a^{[0]}=3,072$ | $0$          |
| CONV1(f=5,s=1) | $(28, 28, 8)$    | $6,272$         | $208$        |
| POOL1          | $(14, 14, 8)$    | $1,568$         | $0$          |
| CONV2(f=5,s=1) | $(10, 10, 16)$   | $1,600$         | $416$        |
| POOL2          | $(5, 5, 16)$     | $400$           | $0$          |
| FC3            | $(120, 1)$       | $120$           | $48,001$     |
| FC4            | $(84, 1)$        | $84$            | $10,081$     |
| Softmax        | $(10, 1)$        | $10$            | $841$             |


## 1.11 Why convolutions?

### Why convolutions

The advantages are *parameter sharing* and *sparsity of connections*.

**Parameter sharing:** A feature detector (such as a vertical edge detector) that's useful in one part of the image is probably useful in another part of the image.

**Sparsity of connections:** In each layer, each output value depends only on a small number of inputs.

Through these two mechanisms, a neural network has a lot fewer parameters, which allows it to be trained with smaller training sets, and it's less prone to be over fitting.

### Putting it together

![[Pasted image 20230717212307.png]]


---

# Case Studies

## 2.1 Why look at case studies?

### Outline

**Classic networks:**
- LeNet-5
- AlexNet
- VGG

**ResNet**

**Inception**


## 2.2 Classic networks

### LeNet-5

>LeCun et al., 1998. Gradient-based learning applied to document recognition

### AlexNet

> Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks

### VGG-16

> Simonyan & Zisserman 2015. Very deep convolutional networks for large-scale image recognition

