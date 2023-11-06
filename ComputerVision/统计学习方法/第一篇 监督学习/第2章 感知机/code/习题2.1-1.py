# 验证感知机为什么不能表示异或

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
y = [-1, 1, 1, -1]
x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)
data = np.c_[x1, x2, y]
data = pd.DataFrame(data, index=None, columns=['x1','x2','y'])
data.head()

positive = data.loc[data['y']==1]
negative = data.loc[data['y']==-1]

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xticks([-0.5, 0, 1, 1.5])
plt.yticks([-0.5, 0, 1, 1.5])

plt.xlabel("x1")
plt.ylabel("x2")

plt.plot(positive['x1'], positive['x2'], "ro")
plt.plot(negative['x1'], negative['x2'], "bx")

plt.legend(['Positive', 'Negative'])
plt.show()