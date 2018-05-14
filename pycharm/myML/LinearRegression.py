import numpy as np
from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        # 截距theta0
        self.intercept_ = None
        # 系数，theta1 ... thetaN
        self.coef_ = None
        # theta列向量
        self._theta = None

    # 根据训练数据集X_train，y_train训练LinearRegression模型
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "特征数据矩阵的行数要等于样本结果数据的行数"

        # 计算X_b矩阵，既将X_train矩阵前面加一列，元素都为一
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # 实现正规方式解
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        # 取到截距和系数
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # 给定待预测数据集X_predict，返回表示X_predict的结果向量
    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
        "截距和系数都不为空，表示已经经过了fit方法"
        assert X_predict.shape[1] == len(self.coef_), \
        "要预测的特征数据集列数要与theta的系数数量相等"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    # 根据测试数据集X_test和y_test确定当前模型的准确度
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"

