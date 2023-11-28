import math
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def plot_converge(clusteres, initial_means):
    plt.cla()  # 清除原有图像

    plt.title("k-meas converge process")
    plt.xlabel("density")
    plt.ylabel("sugar content")

    for cluster in clusteres:
        cluster = np.array(cluster)
        # 画出每个簇的点
        plt.scatter(cluster[:, 0], cluster[:, 1], c="lightcoral")
        # 画出每个簇的凸包
        if len(cluster) == 1:
            continue
        hull = ConvexHull(cluster).vertices.tolist()
        hull.append(hull[0])
        plt.plot(cluster[hull, 0], cluster[hull, 1], "c--")
    # 标记初始中心点
    plt.scatter(initial_means[:, 0], initial_means[:, 1], label="initial center", c="k")
    plt.legend()
    plt.pause(0.5)


def euclidean_distance(data1, data2):
    if len(data1) != len(data2):
        raise Exception("长度不相等无法计算欧氏距离！")
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(data1, data2)]))


def kmeans(file_path, k, initial_points):
    # 读取要聚类的数据
    dataset = np.loadtxt(file_path, delimiter=",")
    # 均值向量
    means = [list(dataset[initial_point - 1]) for initial_point in initial_points]
    initial_means = copy.deepcopy(means)
    means_pre = []
    # 打开交互模式
    plt.ion()
    # 聚类
    while means != means_pre:
        means_pre = copy.deepcopy(means)
        # 保存每簇信息
        clusteres = [[] for _ in range(k)]
        for data in dataset:
            # 与每个均值向量计算距离
            euclidean_distances = [euclidean_distance(data, mean) for mean in means]
            # 归到距离最小的簇
            min_index = euclidean_distances.index(min(euclidean_distances))
            clusteres[min_index].append(data)
        plot_converge(clusteres, np.array(initial_means))
        # 更新均值向量
        means = [list(np.mean(cluster, axis=0)) for cluster in clusteres]
        print("means:{}".format(means))
    # 关闭交互模式
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    file_path = "watermelon4_0_Ch.txt"  # 数据路径
    # k = 3  # 聚类簇数
    # initial_points = [2, 10, 29]  # 初始点
    # k = 4  # 聚类簇数
    # initial_points = [2, 10, 29, 18]  # 初始点
    k = 5  # 聚类簇数
    initial_points = [2, 10, 29, 18, 7]  # 初始点
    kmeans(file_path, k, initial_points)
