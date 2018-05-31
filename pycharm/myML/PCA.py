import numpy as np

class PCA:

    # 初始化PCA
    def __init__(self, n_components):
        assert n_components >= 1, "至少要有一个主成分"
        self.n_components = n_components
        self.component_ = None

    # 训练主成分矩阵
    def fit(self, X, eta=0.01, n_iters=1e4):
        assert self.n_components <= X.shape[1], "主成分数要小于等于样本数据的特征数"

        # 均值归一化
        def demean(X):
            return X - np.mean(X, axis=0)

        # 目标函数
        def f(w, X):
            return np.sum((X.dot(w)**2)) / len(X)
        # 梯度
        def df(w, X):
            return (X.T.dot(X.dot(w)) * 2) / len(X)

        # 求单位向量
        def direction(w):
            return w / np.linalg.norm(w)

        def first_component(X, initial_w, eta=0.01, n_iters=1e4, different=1e-8):

            # 转换初始向量为单位向量，既只表明方向
            w = direction(initial_w)
            cur_iters = 0

            while cur_iters < n_iters:
                # 求出梯度
                gradient = df(w, X)
                # 记录上一个方向向量
                last_w = w
                # 通过梯度上升求下一个方向向量
                w = w + eta * gradient
                # 将新求出的方向向量单位向量化
                w = direction(w)

                if(abs(f(w, X) - f(last_w, X)) < different):
                    break

                cur_iters += 1

            return w

        # 对样本数据的特征数据均值归一化
        X_pca = demean(X)
        # 构建一个空的主成分矩阵，大小和样本数据保持一致
        self.component_ = np.empty(shape=(self.n_components, X.shape[1]))
        for i in range(self.n_components):
            # 随机生成一个初始向量
            initial_w = np.random.random(X_pca.shape[1])
            # 求第一主成分
            w = first_component(X_pca, initial_w, eta, n_iters)
            # 存储主成分
            self.component_[i, :] = w

            X_pca = X_pca - X_pca.dot(w).reshape(-1, 1) * w

        return self

    # 根据主成分矩阵降维样本数据
    def transform(self, X):
        assert X.shape[1] == self.component_.shape[1], "样本数据的列数，既特征数要等于主成分矩阵的列数"

        return X.dot(self.component_.T)

    # 根据主成分矩阵还原样本数据
    def inverse_transform(self, X_pca):
        assert X_pca.shape[1] == self.component_.shape[0], "降维后的样本数据特征数要等于主成分矩阵的行数"

        return X_pca.dot(self.component_)

    def __repr__(self):
        return "PCA(n_components=%d)" % self.n_components
