import numpy as np


class Perceptron(object):
    def __init__(self, eta=0.01, iter=10):
        """
        初始化，eta为变化步长，默认是0.01；iter是学习的轮数，默认是10。
        当然也可不设置迭代次数，改为分类全部正确的那一轮停止迭代。
        """
        self.eta = eta
        self.iter = iter

    def fit(self, X, y):
        """开始学习，每次变更参数"""
        self.w = np.zeros(1 + X.shape[1])  # Add b，wx+b，即w[0]是b，w[1:]是对应的w

        for _ in range(self.iter):
            for xi, target in zip(X, y):
                update = self.eta * (
                    target - self.predict(xi)
                )  # target是目标值，predit计算出来的是预测值，二者差值则是需要变更的量；乘以对应变更步长
                self.w[1:] += update * xi  # 使用的是梯度上升法；梯度上升法和下降法实际是一个公式，应用是上升法是+，下降法是-
                self.w[0] += update
        return self

    def sigmoid(self, x):
        """激活函数sigmoid"""
        return 1 / (1 + np.exp(-x))

    def net_input(self, X):
        """计算wx+b"""
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        """预测结果"""
        return np.where(self.sigmoid(self.net_input(X)) >= 0.5, 1, 0)


if __name__ == "__main__":
    and_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    and_lable = np.array([0, 0, 0, 1])
    ppn_and = Perceptron(eta=0.1, iter=10)
    ppn_and.fit(and_data, and_lable)
    print("----------and----------")
    print("w:{0}, b:{1}".format(ppn_and.w[1:], ppn_and.w[0]))
    for item in and_data:
        print("{0}的预测结果是：{1}".format(item, ppn_and.predict(item)))

    or_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    or_lable = np.array([0, 1, 1, 1])
    ppn_or = Perceptron(eta=0.1, iter=10)
    ppn_or.fit(or_data, or_lable)
    print("----------or----------")
    print("w:{0}, b:{1}".format(ppn_or.w[1:], ppn_or.w[0]))
    for item in or_data:
        print("{0}的预测结果是：{1}".format(item, ppn_or.predict(item)))

