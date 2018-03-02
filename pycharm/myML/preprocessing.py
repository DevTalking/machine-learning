import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scaler_ = None

    # 获取训练数据集的平均值和方差
    def fit(self, X):
        assert X.ndim == 2, "X 的维度必须为2，既X是一个矩阵"

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scaler_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    # 进行均值方差归一化处理
    def transform(self, X):
        assert X.ndim == 2, "X 的维度必须为2，既X是一个矩阵"
        assert self.mean_ is not None and self.scaler_ is not None, "均值和方差不能为空"
        assert X.shape[1] == len(self.mean_), "训练数据集矩阵的列数必须等于均值数组的元素个数"
        assert X.shape[1] == len(self.scaler_), "训练数据集矩阵的列数必须等于方差数组的元素个数"

        X_transform = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            X_transform[:, col] = (X[:, col] - self.mean_[col]) / self.scaler_[col]

        return X_transform