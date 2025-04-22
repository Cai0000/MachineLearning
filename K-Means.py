import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成样本数据
X, _ = make_blobs(n_samples=300, centers=3, random_state=0)

# 模型训练
model = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=11)
model.fit(X)
labels = model.labels_
centers = model.cluster_centers_

# 可视化
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=20)
plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X')
plt.title("K-Means Clustering")
plt.show()