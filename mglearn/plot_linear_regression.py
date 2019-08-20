import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from .datasets import make_wave
from .plot_helpers import cm2


def plot_linear_regression_wave():
    # wave数据集
    X, y = make_wave(n_samples=60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    '''
    z = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])
    z.shape
    (4, 4)

    z.reshape(-1)
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16])

    z.reshape(-1,1)
     array([[ 1],
            [ 2],
            [ 3],
            [ 4],
            [ 5],
            [ 6],
            [ 7],
            [ 8],
            [ 9],
            [10],
            [11],
            [12],
            [13],
            [14],
            [15],
            [16]])
    先前我们不知道z的shape属性是多少，但是想让z变成只有一列，行数不知道多少，
    通过`z.reshape(-1,1)`，Numpy自动计算出有12行，新的数组shape属性为(16, 1)，与原来的(4, 4)配套。
    
    创建100个数据点，在-3和3之间均匀分布
    '''
    line = np.linspace(-3, 3, 100).reshape(-1, 1)

    '''
    构建线性回归模型
    coef_[0],表示w[0]
    intercept_,表示b
    '''
    lr = LinearRegression().fit(X_train, y_train)
    print("w[0]: %f  b: %f" % (lr.coef_[0], lr.intercept_))

    # 新建画布，宽8高8
    plt.figure(figsize=(8, 8))
    # 预测测试数据，画线
    plt.plot(line, lr.predict(line))
    # 绘制训练集圆点
    plt.plot(X, y, 'o', c=cm2(0))
    # 获取当前子图
    ax = plt.gca()
    # 设置坐标轴，左边和下边居中，右边和上边隐藏
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    # 设置y坐标轴范围
    ax.set_ylim(-3, 3)
    # ax.set_xlabel("Feature")
    # ax.set_ylabel("Target")
    # 设置图例
    ax.legend(["model", "training data"], loc="best")
    # 显示网格
    ax.grid(True)
    # 数据和单位缩放相同
    ax.set_aspect('equal')
