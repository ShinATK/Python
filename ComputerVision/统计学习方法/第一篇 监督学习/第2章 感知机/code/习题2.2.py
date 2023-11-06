# 仿照书中第2章感知机学习算法2.1，编写代码并绘制分离超平面

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, X, Y, lr=0.001, plot=True):
        """
        初始化感知机
        :param X: feature vector
        :param Y: label
        :param lr: learning rate
        :param plot: plot or not
        """
        self.X = X
        self.Y = Y
        self.lr = lr
        self.plot = plot
        if plot:
            self.__model_plot = self._ModelPlot(self.X, self.Y)
            self.__model_plot.open_in()
        
    def fit(self):
        weight = np.zeros(self.X.shape[1])
        b = 0
        train_counts = 0

        mistake_flag = True
        while mistake_flag:
            mistake_flag = False
            for index in range(self.X.shape[0]):
                if self.plot:
                    self.__model_plot.plot(weight, b, train_counts)
                loss = self.Y[index] * (weight @ self.X[index] + b)
                if loss <= 0:
                    weight += self.lr*self.Y[index]*self.X[index]
                    b += self.lr*self.Y[index]
                    train_counts += 1
                    print("Epoch {}, weight = {}, b = {}, formula: {}".format(train_counts, weight, b, self.__model_plot.formula(weight, b)))
                    mistake_flag = True
                    break
            if self.plot:
                self.__model_plot.close()
            return weight, b
    class _ModelPlot:
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
        @staticmethod
        def open_in():
            plt.ion()
        
        @staticmethod
        def close():
            plt.ioff()
            plt.show()
        
        def plot(self, weight, b, epoch):
            plt.cla()
            plt.xlim(0, np.max(self.X.T[0]) + 1)
            plt.ylim(0, np.max(self.X.T[1]) + 1)
            scatter = plt.scatter(self.X.T[0], self.X.T[1], c=self.Y)
            plt.legend(*scatter.legend_elements())
            if True in list(weight == 0):
                plt.plot(0, 0)
            else:
                x1 = -b / weight[0]
                x2 = -b / weight[1]
                plt.plot([x1, 0], [0, x2])
                text = self.formula(weight, b)
                plt.text(0.3, x2 - 0.1, text)
            plt.title('Epoch %d' % epoch)
            plt.pause(0.01)
        
        @staticmethod
        def formula(weight, b):
            text = 'x1' if weight[0] == 1 else '%d*x1 ' % weight[0]
            text += '+ x2 ' if weight[1] == 1 else ('+ %d*x2 ' % weight[1] if weight[1] > 0 else '- %d*x2 ' % -weight[1])
            text += '= 0' if b == 0 else ('+ %d = 0' % b if b > 0 else '- %d = 0' % -b)

            return text

X = np.array([[3,3],[4,3],[1,1]])
Y = np.array([1, 1, -1])
model = Perceptron(X, Y, lr=1)
weight, b = model.fit()