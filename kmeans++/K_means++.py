import numpy as np
import matplotlib.pyplot as plt
import datetime

# 读取数据
data = np.genfromtxt('data_0.txt', delimiter=' ')

# 计算欧氏距离
def euclDistance(vector1, vector2):
    return np.sqrt(sum((vector1 - vector2)**2))

def ManhattanDistance(vector1,vector2):
    return np.abs(vector1[0]-vector2[0]) + np.abs(vector1[1]-vector2[1])
time = 0    #记录当前跑到了第几个垂直间隔d   time = 0 对应 d=-6  time = d + 6
IterTimes = 0  #记录迭代次数   是主要的优化指标（时间方面）
KmeansTime = 100 #指调用多少次kmeans函数(避免陷入局部最优)

#修改kmeans还是kmeans++  只是调用初始化质心函数不同 改变即可
# def initCentroids(data, k,times):        
#     numSample, dim = data.shape
#     print("---------------------------------")
#     print(f"({(6+time)*KmeansTime+times}/{8*KmeansTime})距离d={time}, 第{times}次选初始质心")
#     # k个质心
#     centroids = np.zeros((k, dim))
#     # 随机选出k个质心
#     for i in range(k):
#         # 随机选取一个样本的索引
#         index = int(np.random.uniform(0, numSample))
#         # 初始化质心
#         centroids[i, :] = data[index, :]
#     return centroids

# 初始化质心（初始化各个类别的中心点）
def initCentroids(data, k, times):
    numSample,dim = data.shape
    index = np.random.randint(0,numSample)

    centroid = np.array(data[index,:]).reshape(1,-1)   #1维变成2维数组  一行n列
    print("---------------------------------")
    print(f"({(6+time)*KmeansTime+times}/{8*KmeansTime})距离d={time}, 第{times}次选初始质心")
    cnt = 1
    while cnt < k:
        maxDis = 0
        maxDisDataIndex = -1
        for i in range(numSample):
            minDis = 10000

            for j in range(cnt):
                dis_square_ij = np.square(centroid[j,0] - data[i,0]) + np.square(centroid[j,1] - data[i,1]) #第j个质心和第i个样本点之间的距离
                if(minDis > dis_square_ij):
                    minDis = dis_square_ij  #获得了第i个样本点 对所有质心的距离中 最小的距离    我们的目的是找1000个样本点中 这个最小距离的最大值

            if(maxDis < minDis):  #找到这个最小距离最大的样本点
                maxDis = minDis
                maxDisDataIndex = i 
        
        centroid = np.vstack((centroid,data[maxDisDataIndex,:])) #新添质心
        cnt += 1
    return centroid
# k-means算法函数
def kmeans(data, k,times):   #times用于指示当前第几次聚类 打印用
    # 计算样本个数
    numSample = data.shape[0]
    # 保存样品属性（第一列保存该样品属于哪个簇，第二列保存该样品与它所属簇的误差（该样品到质心的距离））
    clusterData = np.array(np.zeros((numSample, 2)))
    # 确定质心是否需要改变
    clusterChanged = True
    # 初始化质心
    centroids = initCentroids(data, k, times)

    cnt = 0
    while clusterChanged:
        cnt += 1
        global IterTimes   #如果没有这行代码  下一句的IterTimes+=1 
        IterTimes += 1   
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

    print(f"迭代{cnt}次后 每个样本所属类不再变化")
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
        plt.plot(centroids[i, 0], centroids[i, 1], mark[7], markersize=20)
    #plt.show()

# 设置k值
k = 6

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

# 画出分类图
def Plot(data, k, centroids, clusterData, pngname):
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
    # 保存图像
    plt.savefig(pngname)
    # 清除图像
    plt.cla()

# 画出方差,均值和标准差图
variances_list = []
means_list = []   #绘图时不用这个 这个没有排序
meanListX = []
meanListY = []  #绘图时用这两个列表， 因为平面图纵坐标只有一个 就分开了x和y
ssum= [] #每个d值计算得到的6簇的总方差
starttime = datetime.datetime.now()

for i in range(-6, 2):
    time = i
    filename = f'data_{i}.txt'
    data = np.genfromtxt(filename, delimiter=' ')

    '''
    优化初始质心(保证质心不会仅仅局部收敛)：
    通过多次随机选择质心，最终选择代价值最小的质心
    '''
    min_loss = 10000
    min_loss_centroids = np.array([])
    min_loss_clusterData = np.array([])
    for j in range(KmeansTime):    #共执行50*8 = 400次
        centroids, clusterData = kmeans(data, k,j)
        loss = np.mean((np.square(clusterData[:, 1])))
        print(f"这一次的损失函数值为{loss}")
        if loss < min_loss:
            min_loss = loss
            min_loss_centroids = centroids
            min_loss_clusterData = clusterData
            print("!!!!!!损失函数值减小  优化质心")
    centroids = min_loss_centroids
    clusterData = min_loss_clusterData
    
    means_list.append(centroids)
    np.savetxt(f"data{i}.txt",clusterData)
    
    pngname = f'png_{i}.png'
    #Plot(data, k, centroids, clusterData, pngname)
    # 计算每个簇误差的方差
    cluster_ids = np.unique(clusterData[:, 0])  # 获取所有簇的唯一标识 返回0，1，2，3，4，5
    variances = []
    for cluster_id in cluster_ids:
        errors = clusterData[clusterData[:, 0] == cluster_id][:, 1]  # 获取属于当前簇的所有误差
        variance = np.mean(np.square(errors)) # 计算误差的方差
        variances.append(variance)
        np.savetxt(f"errors{cluster_id}.txt",errors)
    ssum.append(np.sum(variances)) # 计算方差和
    
    # 将中心点的横坐标进行排序，并根据排序结果对方差进行重新排列
    sorted_indices = np.argsort(centroids[:, 0])
    sorted_centroid = np.array(centroids)[sorted_indices]
    sorted_variances = np.array(variances)[sorted_indices]
    np.savetxt(f"center{i}.txt",sorted_centroid)
    #新添的获得排好序的均值点列表
    meanListX.append(sorted_centroid[:,0])  #sorted_centroid 里面为坐标数据
    meanListY.append(sorted_centroid[:,1])
    variances_list.append(sorted_variances)
    

endtime = datetime.datetime.now()
print(f"一共用时{(endtime - starttime).seconds}.{(endtime - starttime).microseconds//1000}秒")
print(f"一共迭代（kmeans的一次步骤）{IterTimes}次")
#print('variances_list:', variances_list, '\nmeans_list:', means_list)

#设置子图的布局
num_plots = len(variances_list[0])
num_rows = 2  # 子图行数
num_cols = num_plots // num_rows  # 子图列数
if num_plots % num_rows != 0:
    num_cols += 1

#创建画布和子图         ------------------------------------------方差
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))

# 绘制每个子图
for i in range(num_rows):
    for j in range(num_cols):
        idx = i * num_cols + j
        if idx < num_plots:
            ax = axes[i, j] if num_rows > 1 else axes[j]
            ax.plot(range(-6, 2), [v[idx] for v in variances_list], marker='o')
            ax.set_title(f'Variance {idx+1}')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Variance')

# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig('variance_trends.png')

# 显示图像
plt.show()

#------------------------------------------mean
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))

# 绘制每个子图  
for i in range(num_rows):
    for j in range(num_cols):
        ax = axes[i, j] if num_rows > 1 else axes[j]
        rank=i*num_cols+j #重心序号
        for t in range(7): #d数量
            ax.scatter(meanListX[t][rank], meanListY[t][rank], color='red')
            ax.plot([meanListX[t][rank], meanListX[t+1][rank]], [meanListY[t][rank], meanListY[t+1][rank]], color='blue')
        ax.scatter(meanListX[7][rank], meanListY[7][rank], color='black')
        ax.set_title(f'means-{rank+1} change')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-12+rank*5, -2+rank*5)

plt.tight_layout()
plt.axis('equal') 
# 保存图像
plt.savefig('mean_trend.png')
plt.show()

#------------------------------------------meanX
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
for i in range(num_rows):
   for j in range(num_cols):
       idx = i * num_cols + j
       if idx < num_plots:
           ax = axes[i, j] if num_rows > 1 else axes[j]
           ax.plot(range(-6, 2), [v[idx] for v in meanListX], marker='o')
           ax.set_title(f'meanX {idx+1}')
           ax.set_xlabel('Distance')
           ax.set_ylabel('meanX')
#调整子图之间的间距
plt.tight_layout()
plt.axis('equal') 
# 保存图像
plt.savefig('mean_trend_x.png')
plt.show()
# 显示图像 
# ------------------------------------------meanY

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
# 绘制每个子图
for i in range(num_rows):
    for j in range(num_cols):
        idx = i * num_cols + j
        if idx < num_plots:
            ax = axes[i, j] if num_rows > 1 else axes[j]
            ax.plot(range(-6, 2), [v[idx] for v in meanListY], marker='o')
            ax.set_title(f'meanY {idx+1}')
            ax.set_xlabel('Distance')
            ax.set_ylabel('meanY')
# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig('mean_trends_y.png')

# 显示图像
plt.show()
#计算每组中心点的两两距离最大值,得到一般方差
common_v = []
for means in means_list:
    distances = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            distance = euclDistance(means[i], means[j])
            distances.append(distance)
    max_distance = np.max(distances)
    sigma = max_distance*max_distance/(2*k)
    common_v.append(sigma)

# 创建新的图形，并绘制额外数据的图形
plt.figure(figsize=(8, 6))
plt.plot(range(-6, 2), common_v, marker='o', color='red',label='theory')
plt.plot(range(-6, 2), ssum, marker='s', color='blue',label='practice')
plt.xlabel('Distance')
plt.ylabel('Variance')
plt.title(r'the common $\sigma^2$')
plt.grid(True)
plt.legend()
# 保存图像
plt.savefig('the common variance.png')
plt.show()

