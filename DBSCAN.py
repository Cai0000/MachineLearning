from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np

# 生成半月形样本数据（测试噪声鲁棒性）
X, _ = make_moons(n_samples=200, noise=0.05, random_state=41)

# 模型训练（eps: 邻域半径; min_samples: 核心点最小邻域样本数）
dbscan = DBSCAN(eps=0.5, min_samples=5)
predict = dbscan.fit_predict(X)
labels = dbscan.labels_

# 标记噪声点（label=-1）
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# 可视化
unique_labels = set(predict)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # 黑色表示噪声点
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:,0], xy[:,1], c=[tuple(col)], s=20, edgecolors='k')
plt.title("DBSCAN Clustering")
plt.show()