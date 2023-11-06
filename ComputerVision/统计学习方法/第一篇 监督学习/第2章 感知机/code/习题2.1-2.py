from sklearn.linear_model import Perceptron
import numpy as np

X_train = np.array([[1,1],[1,0],[0,1],[0,0]])
y = np.array([-1, 1, 1, -1])

perceptron_model = Perceptron()

perceptron_model.fit(X_train, y)

print("感知机的参数：w=", perceptron_model.coef_[0], "b=", perceptron_model.intercept_[0])