import numpy as np

data = np.array(
    [
        [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1],
        [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9],
    ]
)

data_0 = data[0, :]
data_1 = data[1, :]
print(np.mean(data_0))
print(np.mean(data_1))
data[0, :] = data_0 - np.mean(data_0)
data[1, :] = data_1 - np.mean(data_1)
print(data)

cov_matrix = np.cov(data)
print("协方差矩阵：")
print(cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\n特征值：")
print(eigenvalues)
print("\n特征向量：")
print(eigenvectors)

max_index = np.argmax(eigenvalues)
PCA_result = np.dot(eigenvectors[:, max_index], data)
print("\n降维结果：")
print(PCA_result)
