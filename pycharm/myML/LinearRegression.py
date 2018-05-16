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

    # 使用正规方程解，根据训练数据集X_train，y_train训练LinearRegression模型
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

    # 使用批量梯度下降法，根据训练数据集X_train，y_train训练LinearRegression模型
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], \
            "特征数据矩阵的行数要等于样本结果数据的行数"

        # 定义损失函数
        def L(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        # 定义梯度
        def dL(theta, X_b, y):
            # # 开辟空间，大小为theta向量的大小
            # gradient = np.empty(len(theta))
            # # 第0元素个特殊处理
            # gradient[0] = np.sum(X_b.dot(theta) - y)
            #
            # for i in range(1, len(theta)):
            #     # 矩阵求和可以转换为点乘
            #     gradient[i] = (X_b.dot(theta) - y).dot(X_b[:, i])

            return X_b.T.dot(X_b.dot(theta) - y) * 2 / len(X_b)

        # 实现批量梯度下降法
        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, difference=1e-8):
            theta = initial_theta
            i_iter = 0
            while i_iter < n_iters:
                gradient = dL(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient

                if (abs(L(theta, X_b, y) - L(last_theta, X_b, y)) < difference):
                    break

                i_iter += 1
            return theta

        # 构建X_b
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # 初始化theta向量为元素全为0的向量
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = gradient_descent(X_b, y_train, initial_theta, eta)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    # 使用随机梯度下降法，根据训练数据集X_train，y_train训练LinearRegression模型
    def fit_sgd(self, X_train, y_train, n_iters=5, a=5, b=50):

        assert X_train.shape[0] == y_train.shape[0], \
            "特征数据矩阵的行数要等于样本结果数据的行数"

        assert n_iters >= 1, \
            "至少要搜索一轮"

        # 定义theta查找方向的函数，这里不是全量的X_b矩阵了，而是X_b矩阵中的一行数据，
        # 既其中的的一个样本数据，对应的y值也只有一个
        def dL_sgd(theta, X_b_i, y_i):
            return X_b_i.T.dot(X_b_i.dot(theta) - y_i) * 2

        # 实现随机梯度下降法
        def sgd(X_b, y, initial_theta, n_iters):

            # 定义学习率公式
            def eta(iters):
                return a / (iters + b)

            theta = initial_theta

            # 样本数量
            m = len(X_b)

            # 第一层循环是循环轮数
            for i_inter in range(n_iters):

                # 在每一轮，随机生成一个乱序数组，个数为m
                indexs = np.random.permutation(m)

                # 打乱样本数据
                X_b_new = X_b[indexs]
                y_new = y[indexs]

                # 第二层循环便利所有为乱序的样本数据，既保证样本数据能被随机的，全部的计算到
                for i in range(m):
                    # 每次用一个随机样本数据计算theta搜索方向
                    gradient = dL_sgd(theta, X_b_new[i], y_new[i])
                    # 计算下一个theta
                    theta = theta - eta(i_inter * m + i) * gradient

                return theta

        # 构建X_b
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        # 初始化theta向量为元素全为0的向量
        initial_theta = np.zeros(X_b.shape[1])

        self._theta = sgd(X_b, y_train, initial_theta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]


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

