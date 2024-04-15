import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = np.genfromtxt('data_1.txt', delimiter=' ')

# 计算欧氏距离
def euclDistance(vector1, vector2):
    return np.sqrt(sum((vector1 - vector2)**2))

# 初始化质心（初始化各个类别的中心点）
def initCentroids(data, k):
    numSample, dim = data.shape
    # k个质心
    centroids = np.zeros((k, dim))
    # 随机选出k个质心
    for i in range(k):
        # 随机选取一个样本的索引
        index = int(np.random.uniform(0, numSample))
        # 初始化质心
        centroids[i, :] = data[index, :]
    return centroids

# k-means算法函数
def kmeans(data, k):
    # 计算样本个数
    numSample = data.shape[0]
    # 保存样品属性（第一列保存该样品属于哪个簇，第二列保存该样品与它所属簇的误差（该样品到质心的距离））
    clusterData = np.array(np.zeros((numSample, 2)))
    # 确定质心是否需要改变
    clusterChanged = True
    # 初始化质心
    centroids = initCentroids(data, k)
    while clusterChanged:
        clusterChanged = False
        # 遍历样本
        for i in range(numSample):
            # 该样品所属簇（该样品距离哪个质心最近）
            minIndex = 0
            # 该样品与所属簇之间的距离
            minDis = 100000.0
            # 遍历质心
            for j in range(k):
                # 计算该质心与该样品的距离
                distance = euclDistance(centroids[j, :], data[i, :])
                # 更新最小距离和所属簇
                if distance < minDis:
                    minDis = distance
                    clusterData[i, 1] = minDis
                    minIndex = j
            # 如果该样品所属的簇发生了改变，则更新为最新的簇属性，且判断继续更新簇
            if clusterData[i, 0] != minIndex:
                clusterData[i, 0] = minIndex
                clusterChanged = True
        # 更新质心
        for j in range(k):
            # 获取样本中属于第j个簇的所有样品的索引
            cluster_index = np.nonzero(clusterData[:, 0] == j)
            # 获取样本中于第j个簇的所有样品
            pointsInCluster = data[cluster_index]
            # 重新计算质心(取所有属于该簇样品的按列平均值)
            centroids[j, :] = np.mean(pointsInCluster, axis=0)

    return centroids, clusterData

# 显示分类结果
def showCluster(data, k, centroids, clusterData):
    numSample, dim = data.shape
    # 用不同的颜色和形状来表示各个类别的样本数据
    mark = ['or', 'ob', 'og', 'ok', '+b', 'sb', '<b', 'pb']
    if k > len(mark):
        print('your k is too large！')
        return 1
    # 画样本点
    for i in range(numSample):
        markIndex = int(clusterData[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[markIndex])
    # 用不同的颜色个形状来表示各个类别点（质心）
    mark = ['*r', '*g', '*b', '*k', '+b', 'sb', 'db', '<b', 'pb']
    # 画质心点
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=20)
    plt.show()

# 设置k值
k = 6
'''
优化初始质心(保证质心不会仅仅局部收敛)：
通过多次随机选择质心，最终选择代价值最小的质心
'''
min_loss = 10000
min_loss_centroids = np.array([])
min_loss_clusterData = np.array([])
for i in range(50):   
    centroids, clusterData = kmeans(data, k)    #不同之处在于初始化的质心位置不同  降低陷入局部最优的概率
    loss = sum(clusterData[:, 1]) / data.shape[0]
    if loss < min_loss:
        min_loss = loss
        min_loss_centroids = centroids
        min_loss_clusterData = clusterData
centroids = min_loss_centroids
clusterData = min_loss_clusterData

if np.isnan(centroids).any():
    print("Error")
else:
    print('Cluster complete!')

# 预测函数（因为聚类一旦完成，预测数据所属类别就变成了有监督分类问题，将预测数据归属于距离最近的类别即可，距离采用欧氏距离）
def prediction(datas, k, centroids):
    clusterIndexs = []
    for i in range(len(datas)):
        data = datas[i, :]
        # 处理data数据(处理为可以与质心矩阵做运算的形式)
        data_after = np.tile(data, (k, 1))
        # 计算该点到质心的误差平方（距离）
        distance = (data_after - centroids) ** 2
        # 计算误差平方和
        erroCluster = np.sum(distance, axis=1)
        # 获取最小值所在索引号,即预测x_test对应所属的类别
        clusterIndexs.append([np.argmin(erroCluster)])
    return np.array(clusterIndexs)

# 利用推导式编写预测函数
def predict(datas, k, centroids):
    return np.array([np.argmin(((np.tile(data, (k, 1)) - centroids)**2).sum(axis=1)) for data in datas])

# 画出簇的作用域
# 获取数据值所在的范围
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1

# 生成网格矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
z = predict(np.c_[xx.ravel(), yy.ravel()], k, centroids)
z = z.reshape(xx.shape)
# 绘制等高线图
cs = plt.contourf(xx, yy, z)
# 显示分类结果
showCluster(data, k, centroids, clusterData)

