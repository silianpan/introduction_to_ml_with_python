import numpy as np
import pandas as pd
import os
from scipy import signal
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.datasets import make_blobs

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")


def make_forge():
    # a carefully hand-designed dataset lol
    # make_blobs聚类数据生成器
    # centers：产生数据中心点
    # random_state：随机生成器种子
    # n_samples：数据样本点个数，默认值为100
    # n_features：表示数据的维度，默认为2
    # cluster_std：每个类别的方差，例如我们希望生成2类数据，其中一类比另一类具有更大的方差，可以将cluster_std设置为[1.0,3.0]
    X, y = make_blobs(centers=2, random_state=4, n_samples=30)
    print(X)
    print(y)
    # 将y中7和27位置的点置为0
    y[np.array([7, 27])] = 0
    print(y)
    # 创建长度为X，全为True的数组（ones，全为1，zeros，全为0）
    mask = np.ones(len(X), dtype=np.bool)
    print(mask)
    # 将mask数组中，0，1，5，26位置置为0，即False
    mask[np.array([0, 1, 5, 26])] = 0
    print(mask)
    # 分别在X和y中只去mask中为True的位置的节点
    X, y = X[mask], y[mask]
    print(X)
    print(y)
    return X, y


def make_wave(n_samples=100):
    rnd = np.random.RandomState(42)
    '''
    numpy.random.uniform(low,high,size)
    从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high
    '''
    x = rnd.uniform(-3, 3, size=n_samples)
    print(x)
    y_no_noise = (np.sin(4 * x) + x)
    print(y_no_noise)
    '''
    np.random.normal()
    正态分布
    '''
    y = (y_no_noise + rnd.normal(size=len(x))) / 2
    print(y)
    return x.reshape(-1, 1), y


def load_extended_boston():
    boston = load_boston()
    X = boston.data

    X = MinMaxScaler().fit_transform(boston.data)
    X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    return X, boston.target


def load_citibike():
    data_mine = pd.read_csv(os.path.join(DATA_PATH, "citibike.csv"))
    data_mine['one'] = 1
    data_mine['starttime'] = pd.to_datetime(data_mine.starttime)
    data_starttime = data_mine.set_index("starttime")
    data_resampled = data_starttime.resample("3h").sum().fillna(0)
    return data_resampled.one


def make_signals():
    # fix a random state seed
    rng = np.random.RandomState(42)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    # create three signals
    s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
    s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
    s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

    # concatenate the signals, add noise
    S = np.c_[s1, s2, s3]
    S += 0.2 * rng.normal(size=S.shape)

    S /= S.std(axis=0)  # Standardize data
    S -= S.min()
    return S
