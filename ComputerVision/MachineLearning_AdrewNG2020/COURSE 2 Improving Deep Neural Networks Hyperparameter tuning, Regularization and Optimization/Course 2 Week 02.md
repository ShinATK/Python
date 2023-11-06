# Optimization Algorithms

## 2.1 Mini-batch gradient descent

### Batch vs. mini-batch gradient descent

Vectorization allows you to efficiently compute on $m$ examples.

$$
\begin{align}
\sideset{}{} X_{(n_{x}, m)} &= [x^{(1)}, x^{(2)}, ..., x^{(m)}] \\
\sideset{}{} Y_{(1, m)} &= [y^{(1)}, y^{(2)}, ..., y^{(m)}]
\end{align}
$$

*split up your training sets into smaller training sets*, which is **mini-batch**

What if $m=5,000,000$ ?
	$5,000$ mini-batches of $1,000$ examples each

$$
\begin{align}
\sideset{}{} X_{(n_{x}, m)} &= [\underbrace{x^{(1)} x^{(2)} ... x^{(1000)}}_{X^{\{ 1 \}} (n_{x},1000)} | \underbrace{x^{(1001)} ... x^{(2000)}}_{X^{\{ 2 \}}}| ... | \underbrace{...x^{(m)}}_{X^{\{ t \}}}]
\end{align}
$$
$Y$ the same.

So, mini-batch $t: X^{\{t\}}, Y^{\{ t\}}$

### Mini-batch gradient descent

for t=1, ..., 5000
	Forward propagation on $X^{\{t\}}$
	$Z[1] = W[1]X\{t\} + b[1]$
	$A[1] = g[1](Z[1])$
	 ...
	$A[l] = g[l](Z[l])$
Vectorized implement (1000 examples)
Compute cost  $J=\frac{1}{1000}\sum_{i=1}^{l}L(\hat{y}^{(i)}, y^{(i)})+\frac{\lambda}{2*1000}\sum_l||W^{[l]}||^2_{F}$
Backward propagation to compute gradients cost $J^{\{t\}}$ (using $\ (X^{\{t\}}$, $Y^{\{t\}}$))
	$W^{[l]} := W^{[l]} - \alpha \mathrm{d}W^{[l]}, b^{[l]} := b^{[l]} - \alpha \mathrm{d}b^{[l]}$

**"1 epoch"**, pass through training set, but take 5,000 gradient descent steps.


## 2.2 Understanding mini-batch gradient descent

### Training with mini batch gradient descent

| Batch gradient descent          | Mini-batch gradient descent          |
| ------------------------------- | ------------------------------------ |
| ![[Batch_gradient_descent.png]] | ![[Mini-batch_gradient_descent.png]] |
 
### Choosing your mini-batch size

Two extremes: 
- If mini batch size = $m$ : **Batch gradient desent**, $(X^{\{1\}}, Y^{\{1\}}) = (X, Y)$
	- *Too long per iteration*

- *In practice: Somewhere between in 1 and m*
	- **The fastest learning**
		- Vectorization
		- Make progress without processing entire training set

- If mini batch size = $1$ : **Stochastic gradient descent**, every example is its own mini-batch
	- *Lose speed-up from vectorization*, because of processing a single training example at a time


## 2.3 Exponentially weighted averages

$$V_{t} = \beta V_{t-1} + (1-\beta)\theta_t$$
$V_t$ as approximately averaging over $\frac{1}{1-\beta}$ days temperature.

![[Exponentially_weighted_averages.png]]


## 2.4 Understanding exponentially weighted averages


## 2.5 Bias correction in exponentially weighted average

![[Bias_correction.png]]

*Use $\frac{V_t}{1-\beta^t}$ to calculate $V_t$ on the early stage.*

If you are concerned about the bias during this initial phase, while your exponentially weighted moving average is still warming up. Then bias correction can help you get a better estimate early on.


## 2.6 Gradient descent with momentum

### Gradient descent example

![[Gradient_descent_example.png]]

### Implementation details

On iteration $t$:
	Compute $\mathrm{d}W, \mathrm{d}b$ on the current mini-batch
	$$\begin{align}
		V_{\mathrm{d}W} &= \beta V_{\mathrm{d}W} + (1-\beta)\mathrm{d}W \\
		V_{\mathrm{d}b} &= \beta V_{\mathrm{d}b} + (1-\beta)\mathrm{d}Wb \\
		W &= W - \alpha V_{\mathrm{d}W}, b = b - \alpha V_{\mathrm{d}W}
	\end{align}$$
**Hyperparameters: $\alpha$ $\beta$** 
Usually choose $\beta = 0.9$


## 2.7 RMSprop

**RMSprop**: root mean square prop

```Python
# On the iteration t:
# Compute dW, db on the current mini-batch	

S_dW = beta * S_dW + (1-beta) * dW**2 # element-wise square
S_db = beta * S_db + (1-beta) * db**2 # element-wise square

W = W - alpha * dW/np.sqrt(S_dW)
b = b - alpha * db/np.sqrt(S_db)
```


## 2.8 Adam optimization algorithm

### Adam optimization algorithm

```Python
V_dW, S_dW, V_db, S_db = 0, 0, 0, 0

# On the iteration t:
# Compute dW, db using current mini-batch

# "momentum" beta_1
V_dw = beta_1 * V_dW + (1-beta_1) * dW
V_db = beta_1 * V_db + (1-beta_1) * db

# "RMSprop" beta_2
S_dW = beta_2 * S_dW + (1-beta_2) * dW**2
S_db = beta_2 * S_db + (1-beta_2) * db**2

V_dW_corrected = V_dW / (1-beta_1 ** t)
V_db_corrected = V_db / (1-beta_1 ** t)

S_dW_corrected = S_dW / (1-beta_2 ** t)
S_db_corrected = S_db / (1-beta_2 ** t)

W = W - alpha * V_dW_corrected / (np.sqrt(S_dW_corrected) + varepsilon)
b = b - alpha * V_db_corrected / (np.sqrt(S_db_corrected) + varepsilon)
```

### Hyperparameters choice:

| Hyperparameters | Value                       |
| --------------- | --------------------------- |
| $\alpha$        | needs to be tune            |
| $\beta_1$       | 0.9 for ($\mathrm{d}W$)     |
| $\beta_2$       | 0.999 for ($\mathrm{d}W^2$) |
| $\varepsilon$   | $10^{-8}$                            |


## 2.9 Learning rate decay

Speed up the learning algorithm, is to slowly reduce your learning rate over time.

### Learning rate decay

$$\alpha = \frac{1}{1+decayrate * epochnum} \cdot \alpha_0$$
Suppose $\alpha_{0}=0.2,decayrate=1$

| Epoch | $\alpha$ |
| ----- | -------- |
| 1     | 0.1      |
| 2     | 0.67     |
| 3     | 0.5      |
| 4     | 0.4      |
| ...   | ...         |

### Other learning rate decay methods

**Formula decay**:
Exponential decay
$$\alpha = 0.95^{epochnum} \cdot \alpha_{0}$$
$$\alpha = \frac{k}{\sqrt{epochnum}} \cdot \alpha_{0} \quad or \quad \frac{k}{\sqrt{t}} \cdot \alpha_{0}$$

**Manual decay**


## 2.10 The problem of local optima

### Local optima in neural networks

![](Local-optima.png)

But in fact, a saddle point.

![](Saddle_points.png)

### Problem of plateaus

A plateaus is a place, where the derivative is close to zero for a long time.

![](Problem_plateaus.png)

- Unlikely to get stuck in a bad local optima
- Plateaus can make learning slow

---
