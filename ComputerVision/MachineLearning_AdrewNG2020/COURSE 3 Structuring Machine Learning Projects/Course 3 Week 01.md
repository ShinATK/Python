# Introduction to ML strategy

## 1.1 Why ML Strategy?

![[Motivation_example.png]]


## 1.2 Orthogonalization

Orthogonalization or orthogonality is a system design property that assures that modifying an instruction or a component of an algorithm will not create or propagate side effect to other components of the system. It becomes easier to verify the algorithms independently from one another, it reduces testing and development time.

When a supervised learning system is design, these are the 4 assumptions that needs to be true and orthogonal.

1. Fit training set well on cost function
	if it doesn't fit well, the use of a bigger neural network or switching to a better optimization algorithm might help
2. Fit development set well on cost function
	if it doesn't fit well, regularization or using bigger training set might help
3. Fit test well on cost function
	if it doesn't fit well, the use of a bigger development set might help
4. Performs well in real world
	if it doesn't perform well, the development test set is note correctly or the cost function is not evaluating the right thing.

---

# Setting up your goal

## 1.3 Single number evaluation metric

Set up a single real number evaluation metric for your problem.

### Using a single number evaluation metric

|           | Definition |
| --------- | ---------- |
| Precision | of the examples that your classifier recognizes as cats, what percentage actually are cats           |
| Recall          | of all the images that really are cats, what percentage were correctly recognized by your classifier           |

![[Number_evaluation_metric_example.png]]


## 1.4 Satisficing and optimizing metrics

If there are multiple things you care about by say there's one as the optimizing metric that you want to do as well as possible on and one or more as satisficing metrics were you'll be satifice, almost it does better than some threshold you can now have an almost automatic way of quickly looking at multiple cost size and picking the quote, best one.

Now these evaluation metrics must be evaluated or calculated on a training set or a development set or maybe on the test set.


## 1.5 Train/Dev/Test distributions


## 1.6 Size of dev and test sets

### Old way of splitting data

**When the data is small(100 1,000 10,000):**
70% for train set, 30% for test set
60% for train set, 20% for dev set, 20% for test set

**When the data is big(like 1,000,000):**
98% for train set, (1% for dev set, 1% for test set)less than 20% or 30%

### Size of test set

Set your test set to be big enough to give high confidence in the overall performance of your system.


## 1.7 When to change dev/test sets and metrics


---

# Comparing to human-level performance

## 1.8 Why human-level performance?

### Comparing to human-level performance

![[Comparing_to_human-level_performance.png]]

### Why compare to human-level performance

Humans are quite good at a lot of tasks. So long as ML is worse than humans, you can:

- Get labeled data from humans
- Gain insight from manual error analysis: Why did a person get this right?
- Better analysis of bias/variance


## 1.9 Avoidable bias

### Cat classification example

![[Cat_classification_example.png]]


## 1.10 Understanding human-level performance

### Human-level error as a proxy for Bayes error

![[Pasted image 20230703172426.png]]

To be clear about what your purpose is in defining the term human-level error. And if it is to show that you can surpass a single human and therefore argue for deploying your system in some context, maybe "Typical docotor" is the appropriate definition. But if your goal is the proxy for Bayes error, then "Team of experienced doctors" is the appropriate definition.

### Error analysis example

![[Pasted image 20230703173455.png]]

### Summary of bias/variance with human-level performance

![[Pasted image 20230703173652.png]]


## 1.11 Surpassing human-level performance


## 1.12 Improving your model performance

### The two fundamental assumptions of supervised learning

1. You can fit the training set pretty well. (*Avoidable bias*)
2. The training set performance generalizes pretty well to the dev/test set. (*Variance*)

### Reducing (avoidable) bias and variance


