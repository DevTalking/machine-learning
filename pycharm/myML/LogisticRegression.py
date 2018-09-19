import numpy as np
from .metrics import accuracy_score

class LogisticRegression:

    def __init__(self):
        # 截距theta0
        self.intercept_ = None
        # 系数，theta1 ... thetaN
        self.coef_ = None
        # theta列向量
        self._theta = None

    # 定义Sigmoid私有函数
    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    # 使用批量梯度下降法，根据训练数据集X_train，y_train训练LogisticRegression模型
    def fit(self, X_train, y_train, is_debug=False, eta=0.01, n_iters=1e4):
        assert X_train.shape[0] == y_train.shape[0], \
            "特征数据矩阵的行数要等于样本结果数据的行数"

        # 定义逻辑回归损失函数
        def L(theta, X_b, y):
            # 定义逻辑回归概率公式
            y_hat = self._sigmoid(X_b.dot(theta))

            try:
                return -np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat)) / len(X_b)
            except:
                return float('inf')

        # 定义逻辑回归梯度
        def dL(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        def dL_debug(theta, X_b, y, epsilon=0.01):
            # 开辟大小与theta向量一致的向量空间
            result = np.empty(len(theta))
            # 便利theta向量中的每一个theta
            for i in range(len(theta)):
                # 复制一份theta向量
                theta_1 = theta.copy()
                # 将第i个theta加上一个距离，既求该theta正方向的theta
                theta_1[i] += epsilon
                # 在复制一份theta向量
                theta_2 = theta.copy()
                # 将第i个theta减去同样的距离，既求该theta负方向的theta
                theta_2[i] -= epsilon
                # 求出这两个点连线的斜率，既模拟该theta的导数
                result[i] = (L(theta_1, X_b, y) - L(theta_2, X_b, y)) / (2 * epsilon)
            return result

        # 实现批量梯度下降法
        def gradient_descent(X_b, y, initial_theta, eta, difference=1e-8):
            theta = initial_theta
            i_iter = 0
            while i_iter < n_iters:
                # 当is_debug为True时走debug的求梯度的方法，反之走梯度公式的方法
                if is_debug:
                    gradient = dL_debug(theta, X_b, y)
                else:
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

    # 计算概率，给定待预测数据集X_predict，返回表示X_predict的结果概率向量
    def predict_probability(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
        "截距和系数都不为空，表示已经经过了fit方法"
        assert X_predict.shape[1] == len(self.coef_), \
        "要预测的特征数据集列数要与theta的系数数量相等"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])

        # 返回0，1之间的浮点数
        return self._sigmoid(X_b.dot(self._theta))

    # 给定待预测数据集X_predict，返回表示X_predict的结果向量
    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
        "截距和系数都不为空，表示已经经过了fit方法"
        assert X_predict.shape[1] == len(self.coef_), \
        "要预测的特征数据集列数要与theta的系数数量相等"

        probability = self.predict_probability(X_predict)
        # 将概率转换为0和1的向量，True对应1，False对应0
        return np.array(probability >= 0.5, dtype='int')

    # 根据测试数据集X_test和y_test确定当前模型的准确度
    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"