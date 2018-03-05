import numpy as np

class SimpleLinearRegression:

    def __init__(self):
        self.a_ = None
        self.b_ = None

    # 根据训练数据集x_train和y_train训练简单线性回归模型
    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "简单线性回归只能处理一个样本特征数据，所以x_train必须是一维向量"
        assert len(x_train) == len(y_train), \
            "x_train和y_train的数量必须要对应"

        self.a_ = (np.mean(x_train) * np.mean(y_train) - np.mean(x_train * y_train)) / (np.mean(x_train) ** 2 - np.mean(x_train ** 2))
        self.b_ = np.mean(y_train) - self.a_ * np.mean(x_train)

        return self

    # 给定待预测数据集x_predict，返回预测输出结果向量
    def predict(self, x_predict):
        assert x_predict.ndim == 1, "因为是简单线性回归，所以待预测数据集必须是一维向量"
        assert self.a_ is not None and self.b_ is not None, "必须先执行fit方法计算a和b"

        return np.array([self._predict(x) for x in x_predict])

    # 给定单个待预测数据x_single，返回x_single的预测结果
    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression()"