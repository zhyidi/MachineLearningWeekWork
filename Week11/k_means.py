import math
import numpy as np


def euclidean_distance(data1, data2):
    if len(data1) != len(data2):
        raise Exception("长度不相等无法计算欧氏距离！")
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(data1, data2)]))


def kmeans(file_path, k, initial_points):
    # 保存每簇信息
    clusteres = [[] for _ in range(k)]
    # 读取要聚类的数据
    dataset = np.loadtxt(file_path, delimiter=",")
    # 均值向量
    means = [dataset[initial_point - 1] for initial_point in initial_points]
    means_pre = []
    # 聚类
    while means != means_pre:
        means_pre = means[:]
        for data in dataset:
            # 与每个均值向量计算距离
            euclidean_distances = [euclidean_distance(data, mean) for mean in means]
            # 归到距离最小的簇
            min_index = euclidean_distances.index(min(euclidean_distances))
            clusteres[min_index].append(data)
        # 更新均值向量
        means = [np.mean(cluster, axis=0) for cluster in clusteres]


if __name__ == "__main__":
    file_path = "watermelon4_0_Ch.txt"  # 数据路径
    k = 3  # 聚类簇数
    initial_points = [6, 12, 27]  # 初始点
    kmeans(file_path, k, initial_points)
