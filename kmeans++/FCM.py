import numpy as np
import random
import datetime 
# 加载数据
data = np.loadtxt("data_-6.txt")

# 初始化参数
n_clusters = 6  # 假设我们有6个类别
m = 1.2  # 模糊系数
tolerance = 0.001  # 误差阈值
max_iter = 300  # 最大迭代次数

# 初始化隶属度矩阵U
U = np.random.dirichlet(np.ones(n_clusters), size=len(data))

def update_centroids(data, U, m):
    """
    更新质心位置
    """
    um = U ** m
    return (data.T @ um / np.sum(um, axis=0)).T

def update_membership(data, centroids, m):
    """
    更新隶属度矩阵U
    """
    U_new = np.zeros((len(data), n_clusters))
    for i, x in enumerate(data):
        for j, c in enumerate(centroids):
            sum_j = np.sum([(np.linalg.norm(x - c) / np.linalg.norm(x - c_alt)) ** (2/(m-1)) for c_alt in centroids])
            U_new[i, j] = 1 / sum_j
    return U_new

def fcm(data, n_clusters, m, tolerance, max_iter):
    """
    执行模糊C-均值算法
    """
    U = np.random.dirichlet(np.ones(n_clusters), size=len(data))
    for iteration in range(max_iter):
        centroids = update_centroids(data, U, m)
        U_new = update_membership(data, centroids, m)
        # 检查是否收敛
        if np.linalg.norm(U_new - U) < tolerance:
            break
        U = U_new
    return centroids, U
starttime = datetime.datetime.now()
for i in range(-6, 2):
    filename = f'data_{i}.txt'
    data = np.genfromtxt(filename, delimiter=' ')
    centroids, U = fcm(data, n_clusters, m, tolerance, max_iter)
endtime = datetime.datetime.now()
print(f"一共用时{(endtime - starttime).seconds}.{(endtime - starttime).microseconds//1000}秒")
print("质心:")
print(centroids)
print("隶属度矩阵:")
print(U)
def defuzzify(U):
    """
    对隶属度矩阵U进行去模糊化，以得到确定的分类结果
    """
    labels = np.argmax(U, axis=1)
    return labels

# 使用隶属度矩阵U进行去模糊化
labels = defuzzify(U)

# 打印分类结果
print("分类结果:")
print(labels)
import matplotlib.pyplot as plt

# 假设这里的data和U是之前FCM算法得到的数据点和隶属度矩阵
# data = np.loadtxt("data_0.txt")
# centroids, U = fcm(data, n_clusters, m, tolerance, max_iter)
labels = np.argmax(U, axis=1)

# 绘制散点图
def plot_clusters(data, labels, centroids):
    plt.figure(figsize=(8, 6))
    n_clusters = len(centroids)
    for i in range(n_clusters):
        points = data[labels == i]
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {i}')
        plt.scatter(centroids[i][0], centroids[i][1], marker='*', s=200, c='black')
    plt.title('FCM Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_clusters(data, labels, centroids)