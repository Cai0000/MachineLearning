import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = [[0.06632, 3.1265], [0.4278, 3.1892]]
dataMat = np.array(data)
X = data[:, 0:1]
Y = data[:, 1]

model = LinearRegression(n_jobs=1)
model.fit(X, Y)
print("系数矩阵：", model.coef_)
print("线性回归模型：", model)

predicted = model.predict(X)
plt.scatter(X, Y)
plt.plot(X, predicted, c='r')

plt.xlabel('x')
plt.ylabel('y')