from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成样本数据（假设来自高斯分布）
X, _ = make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.0, 0.5], random_state=42)

# 模型训练
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(X)
labels = gmm.predict(X)

# 获取均值和协方差矩阵
means = gmm.means_
covariances = gmm.covariances_

# 可视化
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=20)
plt.scatter(means[:,0], means[:,1], c='red', s=200, marker='X')
plt.title("GMM Clustering")
plt.show()