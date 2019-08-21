import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs

from .plot_helpers import discrete_scatter


def plot_linear_svc_regularization():
    # make_blobs聚类数据生成器
    # centers：产生数据中心点
    # random_state：随机生成器种子
    # n_samples：数据样本点个数，默认值为100
    # n_features：表示数据的维度，默认为2
    # cluster_std：每个类别的方差，例如我们希望生成2类数据，其中一类比另一类具有更大的方差，可以将cluster_std设置为[1.0,3.0]
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # a carefully hand-designed dataset lol
    y[7] = 0
    y[27] = 0
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    for ax, C in zip(axes, [1e-2, 10, 1e3]):
        # 离散方法
        discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)

        # 创建线性支持向量
        svm = LinearSVC(C=C, tol=0.00001, dual=False).fit(X, y)
        # w系数
        w = svm.coef_[0]
        a = -w[0] / w[1]
        # 在6和13之间均匀分布
        xx = np.linspace(6, 13)
        yy = a * xx - (svm.intercept_[0]) / w[1]
        ax.plot(xx, yy, c='k')
        # x轴范围
        ax.set_xlim(x_min, x_max)
        # y轴范围
        ax.set_ylim(y_min, y_max)
        # 坐标轴刻度
        ax.set_xticks(())
        ax.set_yticks(())
        # 标题
        ax.set_title("C = %f" % C)
    axes[0].legend(loc="best")

if __name__ == "__main__":
    plot_linear_svc_regularization()
    plt.show()
