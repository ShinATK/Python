# Setting up your ML application

## 1.1 Train/dev/test sets

### Applied ML is a highly iterative process

Parameters:
	# layers
	# hidden units
	learning rates
	activation functions
	...

### Mismatched train/set distribution

Train set:
	Cat pictures from webpages

Dev/test set:
	Cat pictures from users using your app

The two sets may be different.
So need to *make sure dev and test come from same distribution*
Not haveing a test set might be okay. (Only dev set)


## 1.2 Bias/Variance

![](highbias_justright_highvariance.png)

Two key numbers to look at to understand bias and variance will be:
- Traing set error
- Dev set error

*Here, consider human error $\approx 0$, which is the* **base error**

| Train set error | Dev set error | case                        |
| --------------- | ------------- | --------------------------- |
| 1%              | 11%           | *high variance*             |
| 15%             | 16%           | *high bias*                 |
| 15%             | 30%           | *high bias & high variance* |
| 0.5%            | 1%            | **Low bias & Low variance**                            |


## 1.3 Basic "recipe" for machine learning


---

# Regularizing your neural network

## 1.4 Regularization

to prevent *over fitting* which is *high variance*

### Logistic regression
$\min_{w,b}J(w,b)$
$$J(w,b) = \frac{1}{m}\sum {L({\hat{y}}^{(i)}, y^{(i)})}$$
After regularization:
$$J(w,b) = \frac{1}{m}\sum {L({\hat{y}}^{(i)}, y^{(i)})}+\frac{\lambda}{2m}||w||^2_2$$
the norm of $w$ squared is just equal to: $||w||^2_2 = \sum^{n_x}_{j=1}{w_j^2} = w^T w$
This is called **L2 Regularization**.

### L1 Regularization
$$\frac{\lambda}{m}\sum{|w|} = \frac{\lambda}{2m}{||w||_1}$$
and $w$ will end up being sparse, which means $w$ will have a lot of zeros in it.

$\lambda$ *regularization parameter*, this is another hyper parameter that you might have to tune. When set this hyper parameter in Python, use `lambd` to not clash with the reserved keyword in Python.

### Neural network

$$J(w^{[1]}, b^{[1]},..., w^{[l]}, b^{[l]}) = \frac{1}{m}\sum^m_{i=1}{L({\hat{y}}^{(i)}), y^{(i)}} + \frac{\lambda}{2m} \sum^L_{l=1}{||w^{[L]}||^2_F}$$
and $||w^{[L]}||^2_F = \sum^{n^{[l-1]}}_{i=1}{\sum^{n^{[l]}}_{j=1}{(w^{[l]})^2_{ij}}}$

$$dw^{[l]} = (from \quad backprop) + \frac{\lambda}{m}w^{[l]}$$
so *with L2 regularization*, $w$ update: $$w^{[l]} := w^{[l]} - dw^{[l]} = (1-\frac{\alpha \lambda}{m})w-\alpha(from \quad backprop)$$
**L2 Regularization** also called *weight decay*


## 1.5 Why regularization reduces overfitting

![](highbias_justright_highvariance.png)

When *high bias*, the neural network needs more **Nonlinearity**, but when *high variance*, the neural network needs more **Linearity**.

And we assumed the activation *ReLU*,
So, more **Nonlinearity** means less $\lambda$ as well as bigger $w$, and it means in the **Nonlineartiy** area of activation function *ReLU*.
And bigger $\lambda$ means less $w$, in the **Linearity** area of activation function *ReLU*.

So, *high bias* needs more **Nonlinearity** and *high variance* needs more **Linearity**.


## 1.6 Dropout regularization

In addition to **L2 regularization**, another very powerful regularization techniques is called **dropout**.

Assumed that a neural network like the picture is trained, and there is *overfitting*.

![](assumed_overfitting_neuralnetwork.png)

### Droupout

With dropout, what we are going to do is *go through each of the layers* of the network and *set some probability of eliminating a node* in neural network.

### Implementing dropout ("Inverted dropout")

Illustrate with layer $l=3$.
```python
keep_prob = 0.8 # the probability that a given hidden unit will be kept
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3) # a3 *= d3, every elements operation
a3 /= keep_prob # scale a3, to keep the expected value of a3
```

### Making predictions at test time

*Not to use dropout at test time, in particular:*
$$
\begin{align}
	z^{[1]} &= w^{[1]} a^{[0]} + b^{[1]} \\\
	a^{[1]} &= g^{[0]}(z^{[1]}) \\
	z^{[2]} &= w^{[2]} a^{[1]} + b^{[2]} \\
	a^{[2]} &= ... \\
	\hat{y} &= ...
\end{align}
$$


## 1.7 Understanding dropout

### Why does drop-out work?

Intuition: *Can't rely on any one feature, so have to spread out weights.*
	Each node can't just rely on any one input feature, beacause any input features maybe randomly *drop-out*, it has to *spread out weights to each unit that link to itself*.
	Dop-out has a similar effect to L2 regularization. Only to L2 regularization applied to different ways can be a little bit different and even more adaptive to the scale of different inputs.
	It is also feasible to *vary* `keep_prob` *by layer*.


## 1.8 Other regularization methods

### Data augmentation

### Early stopping

---

# Setting up your optimization problem

## 1.9 Normalizing inputs

### Normalizing training sets

0. Original data set
1. Subtract out or zero out the mean
$$
\begin{align}
	\mu &= \frac{1}{m}\sum^{m}_{i=1}x^{(i)} \\
	x &:= x - \mu
\end{align}
$$
2. Normalize the variances
$$
\begin{align}
	\sigma^2 &= \frac{1}{m}\sum^{m}_{i=1}{{x^{(i)}}^2} \\
	x &/= \sigma^2
\end{align}
$$

| Original data set                | Subtract out/Zero out        | Normalize the variances |
| -------------------------------- | ---------------------------- | ----------------------- |
| ![](original_data_set.png) | ![](substract_out.png) |  ![](normalize_the_variances.png)                       |

### Why normalize inputs?
$$J(w,b) = \frac{1}{m}\sum^{m}_{i=1}{L({\hat{y}}^{(i)},y^{(i)})}$$

| Unnormalized          | Normalized          |
| --------------------- | ------------------- |
| ![](unnormalized.png) | ![](normalized.png) |


## 1.10 Vanishing/exploding gradients

When you are training a very deep network, your derivatives or your slopes can be sometimes get either very big or very small, maybe even exponentially small, makes training difficult.

### Vanishing/exploding gradients

![](vanishing_exploding_gradients.png)

Assumed that in this deep network $b^{[l]} = 0$, then $$y = w^{[l]}w^{[l-1]}w^{[l-2]}...w^{[1]} \cdot x$$
where, $a[i] = w^{[i-1]}...w^{[1]} \cdot x$
If
$$
w^{[l]} = 
\begin{bmatrix}
	1.5 & 0 \\
	0 & 1.5
\end{bmatrix}
$$
then,
$$
\hat{y} = w^{[l]}
\begin{bmatrix}
	1.5 & 0 \\
	0 & 1.5 
\end{bmatrix}^{l-1}
\cdot x
$$

So, in deep neural network the value of $y$ will explode.


## 1.11 Weight initialization for deep networks

### Single neuron example

Assume a single neuron example like this with four input features.

![](single_neuron_example.png)

In this network, $z = w_{1}x_{1}+w_{2}x_{2}+... +w_{n}x_{n} +b$
So, we hope larger $n$ gets smaller $w_{i}$, then in Python

While activation function is *ReLU*
```python
w[l] = np.random.randn(shape) * np.sqrt(2/n[l-1])
```

For *tanh* activation function: **Xavier initialization**
```python
w[l] = np.random.randn(shape) * np.sqrt(1/n[l-1])
```

Other version:
```python
w[l] = np.random.randn(shape) * np.sqrt(2/(n[l-1]+n[l]))
```


## 1.12 Numerical approximation of gradients

### Checking your derivative computation


## 1.13 Gradient Checking

### Gradient check for a neural network

Take $W^{[1]}, b^{[1]},...,W^{[l]}, b^{[l]}$ and reshape into a big vector $\theta$.

Take $\mathrm{d} W^{[1]}, \mathrm{d}b^{[1]},..., \mathrm{d}W^{[l]}, \mathrm{d}b^{[l]}$ and reshape into a big vector $\mathrm{d} \theta$.

### Gradient checking (Grad check)

$$
\begin{align}
for \quad each \quad i: &\\
	\mathrm{d} \theta_{approx}[i] &= \frac{J(\theta_1, \theta_2,..., \theta_i+\varepsilon,...) - J(\theta_1, \theta_2,..., \theta_i-\varepsilon,...)}{2 \cdot \varepsilon} \\
	\mathrm{d}\theta[i] &= \frac{\partial J}{\partial \theta_i}
\end{align}	
$$
What you want to do is check if $\mathrm{d}\theta_{approx} \approx \mathrm{d}\theta$.

Check methods: $$\frac{||\mathrm{d}\theta_{approx} - \mathrm{d}\theta||_2}{||\mathrm{d}\theta_{approx}||_{2} + ||\mathrm{d}\theta||_2}$$
The smaller, the better.


## 1.14 Gradient Checking implementation notes

- Don't use in training - *Only to debug*
- If algorithm fails grad check, look at components to try to identify bug
- Remeber regularization
- Doesn't work with dropout (dropout will randomly eliminate different subsets of the hidden units)
- Run at random initialization; perhaps again after some training

---
