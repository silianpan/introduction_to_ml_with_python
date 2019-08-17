import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

from .datasets import make_forge
from .plot_helpers import discrete_scatter


def plot_knn_classification(n_neighbors=1):
    # forge数据集
    X, y = make_forge()

    # 测试集
    X_test = np.array([[8.2, 3.66214339], [9.9, 3.2], [11.2, .5]])
    # euclidean_distances计算向量之间的距离，训练集和测试集之间的距离
    dist = euclidean_distances(X, X_test)
    '''
    argsort数组值从小到大的索引值
    axis=0：按列排序；axis=1：按行排序
    例1：>>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0]) 
    例2：>>> x = np.array([[0, 3], [2, 2]])
    >>> np.argsort(x, axis=0)
    array([[0, 1],[1, 0]])
    '''
    closest = np.argsort(dist, axis=0)
    print(closest)
    print(closest.T)

    # T行列转置
    for x, neighbors in zip(X_test, closest.T):
        for neighbor in neighbors[:n_neighbors]:
            plt.arrow(x[0], x[1], X[neighbor, 0] - x[0],
                      X[neighbor, 1] - x[1], head_width=0, fc='k', ec='k')

    # k近邻分类，fit方法为训练方法，构建模型
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    # 离散点图形，clf.predict(X_test)，测试集标签
    test_points = discrete_scatter(X_test[:, 0], X_test[:, 1], clf.predict(X_test), markers="*")
    training_points = discrete_scatter(X[:, 0], X[:, 1], y)
    plt.legend(training_points + test_points, ["training class 0", "training class 1",
                                               "test pred 0", "test pred 1"])
