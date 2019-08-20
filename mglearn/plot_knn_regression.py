import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import euclidean_distances

from .datasets import make_wave
from .plot_helpers import cm3


def plot_knn_regression(n_neighbors=1):
    # wave数据集
    X, y = make_wave(n_samples=40)
    # 测试集
    X_test = np.array([[-1.5], [0.9], [1.5]])

    # euclidean_distances计算向量之间的距离，训练集和测试集之间的距离
    # 欧式距离
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

    # 创建画布大小为宽为10，高为6英寸的图像
    plt.figure(figsize=(10, 6))

    # 构建模型
    reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)
    # 对测试数据进行预测
    y_pred = reg.predict(X_test)

    # 绘制箭头
    for x, y_, neighbors in zip(X_test, y_pred, closest.T):
        for neighbor in neighbors[:n_neighbors]:
            '''
            从(x, y)到(x+dx, y+dy)绘制箭头
            (x, y)起始点的坐标
            dx,dy: 分别沿x和y方向上的箭头长度
            head_width: 箭头的宽度
            edgecolor或者ec: 颜色或无或'自动'
            facecolor或fc: 颜色或无
            C = Cyan(青色） M = Magenta（洋红或品红） Y = Yellow（黄色） K = blacK（黑色）
            '''
            plt.arrow(x[0], y_, X[neighbor, 0] - x[0], y[neighbor] - y_,
                      head_width=0, fc='k', ec='k')

    # plt.plot表示绘制标记
    # 绘制圆圈作为训练集效果
    # c是color，颜色；markersize表示标记大小；第三个参数'o'表示标记
    train, = plt.plot(X, y, 'o', c=cm3(0))
    # 绘制五角星作为测试集
    test, = plt.plot(X_test, -3 * np.ones(len(X_test)), '*', c=cm3(2),
                     markersize=20)
    # 绘制五角星作为测试集效果
    pred, = plt.plot(X_test, y_pred, '*', c=cm3(0), markersize=20)
    # 绘制测试集垂直线类型
    plt.vlines(X_test, -3.1, 3.1, linestyle="--")
    # 绘制图例
    plt.legend([train, test, pred],
               ["training data/target", "test data", "test prediction"],
               ncol=3, loc=(.1, 1.025))
    # y坐标范围
    plt.ylim(-3.1, 3.1)
    # x轴标签
    plt.xlabel("Feature")
    # y轴标签
    plt.ylabel("Target")
