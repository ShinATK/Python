# Error Analysis

## 2.1 Carrying out error analysis

### Look at dev examples to evaluate ideas

![[Pasted image 20230705102315.png]]

### Evaluate multiple ideas in parallel

![[Pasted image 20230705104701.png]]

In summary, to carry out error analysis, you should find a set of mislabeled examples, either in your dev set, or in your development set. And look at the mislabeled examples for false positives and false negatives. And just count up the number of errors that fall into various different categories. During this process, you might be inspired to generate new categories of errors. Create new categories during that process. But by counting up the fraction of examples that are mislabeled in different ways, often this will help you prioritize or give you inspiration for new directions to go in.


## 2.2 Cleaning up Incorrectly labeled data


## 2.3 Build your first system quickly, then iterate


---

# Mismatched training and dev/test data

## 2.4 Training and testing on different distributions


## 2.5 Bias and Variance with mismatched data distributions


## 2.6 Addressing data mismatch


---

# Learning from multiple tasks

## 2.7 Transfer learning

### Transfer learning

![[Pasted image 20230705164418.png]]

### When transfer learning makes sense

- Task A and B have the same input x.
- You have a lot more data for Task A than Task B.
- Low level features from A could be helpful for learning B.

## 2.8 Multi-task learning

In multi-task learning, you start off simultaneously, trying to have one neural network do several things at the same time. And each of these task helps hopefully all of the other task.

### Simplified autonomous driving example

![[Pasted image 20230705163229.png]]

### Neural network architecture

![[Pasted image 20230705162958.png]]

And the main difference compared to the eariler finding cat classification examples is that you're now summing over j equals 1 through 4.

ANd the main difference between this and softmax regression, is that unlike softmax regression, which assigned a single label to single example. This one image can have multiple labels.

If some of the eariler features in neural network can be shared between these different types of objects, then you find that training one neural network to do four things results in better performance than training four completely separate neural networks to do the four tasks separately.

### When multi-task learning makes sense

- Training on a set of tasks that could benefit from having shared lower-level features.
	For the autonomous driving example, it makes sense that recognizing traffic lights and cars and pedestrains, those should have similar features that could also help you recognize stop signs, because these are all features of roads.
- Usually: Amount of data you have for each task is quite similar.
- Can train a big enough neural network to do well on all the tasks.

---

# End-to-end deep learning

## 2.9 What is end-to-end deep learning

### What is end-to-end learning?

Speech recognition example

![[Pasted image 20230706100940.png]]

According to the size of the data set, 
	large: the end-to-end approach
	medium: intermediate approach
	small: the more traditional pipeline approach

### Face recognition

 ![[Pasted image 20230706103526.png]]

Use two steps:
- each of the two problems you're solving is actually much simpler 
- you have a lot of data for each of the two sub-tasks

SO because you don't hav enough data to solve this end-to-end learning problem, but you do have enough data to solve sub-problems one and two, in practice, breaking this down to two-sub-problems results in better performance than a pure end-to-end deep learning approach.

### More examples

![[Pasted image 20230706104006.png]]


## 2.10 Whether to use end-to-end learning

### Pros and cons of end-to-end deep learning

Pros:
- *Let the data speak*. So if you have enough x,y data then whatever is the most appropriate function mapping from x to y, if you train a big enough neural network, hopefully the neural network will figure it out. And by having a pure machine learning approach, your neural network learning input from x to y maybe more able to capture whatever statistic are in the data, rather than being forced to reflect human preconceptions.
- *Less hand-designing of components needed*.

Cons:
- *May need large amount of data*.
- *Excludes potentially useful hand-designed components*.

### Applying end-to-end deep learning

Key question: **Do you have sufficient data to learn a function of the complexity needed to map x to y?**