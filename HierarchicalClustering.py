import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# 生成样本数据
X, _ = make_blobs(n_samples=50, centers=3, random_state=42)

# 计算层次聚类链接矩阵（使用平均连接）
Z = linkage(X, method='average')

# 树状图可视化
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.xticks(rotation=90)
plt.show()

from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, linkage='average')
labels = model.fit_predict(X)