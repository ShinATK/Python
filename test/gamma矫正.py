import numpy as np
import matplotlib.pyplot as plt

def Gamma(x, gamma=2.2):
    return x ** gamma

def Gamma2(x, gamma=0.45):
    return x ** gamma


x = np.arange(0, 1, 0.01)

plt.plot(x, Gamma(x), label="gamma=2.2")
plt.plot(x, Gamma2(x), label="gamma=0.45")

plt.plot(x, Gamma(Gamma2(x)), label="gamma correction")

plt.legend()
plt.show()

