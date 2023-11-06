# Hyperparameter tuning

## 3.1 Tuning process

How do you go about finding a good setting for these hyperparameters?

### Hyperparameters

$\alpha$ $\beta\approx0.9$
$\beta_{1}=0.9$ $\beta_{2}=0.999$ $\varepsilon=10^{-8}$ 
$\#layers$
$\#hidden units$
leanring rate decay
mini-batch size

### Try random values: Dont't use a grid

| Grid                                              | Random                                              |
| ------------------------------------------------- | --------------------------------------------------- |
| ![](Grid_hyperparameters.png) | ![](Random_hyperparameters.png) |

**Use random point**

### Coarse to fine

![](Coarse_to_fine.png)

**Use a coarse to fine sampling scheme**
Find the point that works the best and some other points around it tended to work really well, then you zoom in to a smaller region of the hyperparameters and then sample more density within this space.


## 3.2 Using an appropriate scale to pick hyperparameters

Question: sampling at random doesn't mean sampling uniform at random over the range of vaild values.

### Picking hyperparameters at random

Randomly picking might be a reasonable thing to do for some hyperparameters, but this is not true for all hyperparameters.

### Appropriate scale for hyperparameters

For example,
learning rate $\alpha=0.0001,...,1$
If sample values uniformly at random over this number line, about 90% of the values you sample would be between 0.1 and 1. So you're using 90% of the resources to search between 0.1 and 1, and only 10% of the resources to search between 0.0001 and 0.1. 

So instead of using a linear scale, it seems more reasonable to search for hyperparameters on a **log scale**.

![[linearscale_logscale.png]]

In python:
```python
r = -4 * np.random.rand()
alpha = 10**r
```

### Hyperparameters for exponentially weighted averages

![[Pasted image 20230701095731.png]]


## 3.3 Hyperparameters tuning in practice: Pandas vs. Caviar

### Re-test hyperparameters occasionally

![[Pasted image 20230701100625.png]]

![[Pasted image 20230701100603.png]]

---

# Batch Normalization

## 3.4 Normalizing activations in a network

### Normalizing inputs to speed up learning

![[Pasted image 20230701101034.png]]

### Implementing Batch Norm

![[Pasted image 20230701102744.png]]


## 3.5 Fitting Batch Norm into a neural network

### Adding Batch Norm to a network

![[Pasted image 20230701104216.png]]

### Working with mini-batches

![[Pasted image 20230701105435.png]]

### Implementing gradient descent

![[Pasted image 20230701105754.png]]


## 3.6 Why does Batch Norm work?

### Learning on shifting input distribution

It makes weights, later or deeper than your network, say the weight on layer 10, more robust to changes to weights in earlier layers of the neural network.

But when the input "x" and "y" change, which means the data distribution changed, then the network may not work. It is called "covariate shift"

![[Pasted image 20230701161021.png]]

### Why this is a problem with neural networks?

From the perspective of this third hidden layer, it gets some values, called $a^{[2]}_{1}, a^{[2]}_{2}, a^{[2]}_{3}, a^{[2]}_{4}$.

And the job of the third hidden layer is to take these values and find a way to map them to $\hat{y}$.

In fact, before these values $a^{[2]}_{1}, a^{[2]}_{2}, a^{[2]}_{3}, a^{[2]}_{4}$, there are $w^{[2]}, b^{[2]}$ and so on, if these values change, then the value $a^{[2]}$ will also change. 

![[Pasted image 20230701161120.png]]

**So from the perspective of this third hidden layer, these hidden unit values are changing all the time, and so it's suffering from the problem of covariate shift.**

What Batch Norm does is:
	It **reduces the amount** that the distribution of these hidden unit values shifts around.
The values for $z^{[2]}_{1}, z^{[2]}_{2}$ can change when the neural network updates the parameters in the earlier layers. *But what batch norm ensures is that no matter how it changes, the mean and the variance of $z^{[2]}_{1}, z^{[2]}_{2}$ will remain the same, or whatever value is governed by $\beta^{[2]}, \gamma^{[2]}$*

It limits the amount to which updating the parameters in the earlier layers can affect the distribution of values that the third layer now sees and therefore has to learn on. And so batch norm reduces the problem of the input values changing, it really causes these values to become more stable, so that the later layers of the neural network has more firm ground to stand on.

### Batch Norm as regularization

- Each mini-batch is scaled by the mean/variance computed on just that mini-batch
- This adds some noise to the values $z^{[l]}$ within that minibatch. So similar to dropout, it adds some noise to each hidden layer's activations.
- This has a slight regularization effect.

Batch Norm handles data one mini-batch at a time.


## 3.7 Batch Norm at test time

During training time, Batch Norm processes your data one mini batch at a time, but the test time you might need to process the examples one at a time.

### Batch Norm at test time

$$\mu = \frac{1}{m} \sum_{i} z^{(i)}$$
$$\sigma^{2}= \frac{1}{m} \sum_{i}(z^{(i)} - \mu)^2$$
$$z^{(i)}_{norm} = \frac{z^{(i)}-\mu}{\sqrt{\sigma^{2}+ \varepsilon}}$$
$$\tilde{z}^{(i)} = \gamma z^{(i)}_{norm} + \beta$$
So, notice that $\mu$ and $\sigma^2$ which you need for this scaling calculation are computed on the entire mini batch. But the test time you might not have a mini batch of 64 or 56 examples to process at the same time. So, you need some different way of coming up with $\mu$ and $\sigma^2$.

![[Pasted image 20230702145704.png]]

During training time, $\mu$ and $\sigma^2$ are computed on an entire mini batch of say, 64, 28 or some number of examples. But at test time, you might need to process a single example at a time. So the way to do that is to estimate $\mu$ and $\sigma^2$ from your training set, and implement an exponentially weighted average.

---

# Multi-class classification

## 3.8 Softmax regression

What if we have multiple possible classes?

### Recognizing cats, dogs, and baby chicks

![[Pasted image 20230702150512.png]]

### Softmax layer

![[Pasted image 20230702154608.png]]

### Softmax examples

![[Pasted image 20230702154805.png]]


## 3.9 Training a softmax classifer

### Understanding softmax

![[Pasted image 20230702155313.png]]

### Loss function

![[Pasted image 20230702155946.png]]

### Gradient descent with softmax

![[Pasted image 20230702155922.png]]

---

# Programming Frameworks

## 3.10 Deep Learning frameworks

![[Pasted image 20230702160504.png]]


## 3.11 Programming Frameworks: TensorFlow

```python
import numpy as np
import tensorflow as tf

coefficients = np.array([[1.], [-10.], [25]])

w = tf.Variable(0, dtype=tf.float32) # parameter
x = tf.placeholder(tf.float32, [3,1]) # data
# cost = tf.add(tf.add(w**2, tf.multiply(-10, w)), 25)
# cost = w**2 - 10*w + 25
csot = x[0][0]*w**2 + x[1][0]*w + x[2][0] # (w-5)**2
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

session = tf.Session()
session.run(init)
print(session.run(w)) # 0.0
# with command
# with tf.Session() as session:
# 	session.run(init)
# 	print(session.run(w))

session.run(train, feed_dict={x:coefficients})
print(session.run(w) # 0.1

for i in range(1000):
	session.run(train, feed_dict={x:coefficients})
print(session.run(w)) # 4.99999
```

---
