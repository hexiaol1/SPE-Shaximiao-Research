import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 读取数据（跳过第一行标题）
data = np.loadtxt("D:\\repository\\spe\\data\\LOG(FZI).txt", skiprows=1)
X = data.reshape(-1, 1)

# 计算 SSE
sse = []
K = range(1, 16)

for k in K:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    centers = gmm.means_.flatten()
    labels = gmm.predict(X)
    sse_k = np.sum((X.flatten() - centers[labels]) ** 2)
    sse.append(sse_k)

# 绘制结果
plt.figure(figsize=(7, 5))
plt.plot(K, sse, marker='o')
plt.title("SSE随类别数变化 (GMM聚类)")
plt.xlabel("类别数 (k)")
plt.ylabel("SSE（误差平方和）")
plt.grid(True)
plt.show()
