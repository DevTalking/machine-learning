import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

def kNNClassify(k, XTrain, yTrain, x):

    assert 1 <= k <= XTrain.shape[0], "k 的取值范围不正确"
    assert XTrain.shape[0] == yTrain.shape[0], "训练样本数据行数应该与训练结果集行数相同"
    assert XTrain.shape[1] == x.shape[0], "训练样本数据特性个数应该与被预测数据特性个数相同"

    distances = [sqrt(np.sum((xTrain - x) ** 2)) for xTrain in XTrain]
    nearest = np.argsort(distances)

    topKy = [yTrain[i] for i in nearest[:k]]
    votes = Counter(topKy)

    return votes.most_common(1)[0][0]

class KNNClassifier:

    # 初始化kNN分类器
    def __init__(self, k):
        assert k >= 1, "k 值不能小于1"

        self.k = k
        self._XTrain = None
        self._yTrain = None

    # 根据训练数据集XTrain和yTrain训练kNN分类器，在kNN中这一步就是复制训练数据集
    def fit(self, XTrain, yTrain):
        assert XTrain.shape[0] == yTrain.shape[0], \
            "训练样本特征数据集的行数要与训练样本分类结果数据集的行数相同"
        assert XTrain.shape[0] >= self.k, \
            "训练样本特征数据集的行数，既样本点的数量要大于等于k值"

        self._XTrain = XTrain
        self._yTrain = yTrain
        return self

    # 输入样本数据，根据模型进行预测
    def predict(self, XPredict):
        assert self._XTrain is not None and self._yTrain is not None, \
            "在执行predict方法前必须先执行fit方法"
        assert XPredict.shape[1] == self._XTrain.shape[1], \
            "被预测数据集的特征数，既列数必须与模型数据集中的特征数相同"

        ypredict = [self._predict(x) for x in XPredict]
        return np.array(ypredict)

    # 实现私有的预测方法，kNN算法的核心代码
    def _predict(self, x):
        assert x.shape[0] == self._XTrain.shape[1], \
            "输入的样本数据的特征数量必须等于模型数据，既训练样本数据的特征数量"

        distance = [sqrt(np.sum((xTrain - x) ** 2)) for xTrain in self._XTrain]
        nearest = np.argsort(distance)
        topK = [self._yTrain[i] for i in nearest[:self.k]]
        votes = Counter(topK)

        return votes.most_common(1)[0][0]

    # 根据测试数据集得出模型的准确率
    def accuracy_score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_predict, y_test)


    def __repr__(self):
        return "kNN(k=%d)" % self.k
