import numpy as np
from math import sqrt
from collections import Counter

def kNNClassify(k, XTrain, yTrain, x):
    assert 1 <= k <= XTrain.shape[0], "k 的取值范围不正确"
    assert XTrain.shape[0] == yTrain.shape[0], "训练样本数据行数应该与训练结果集行数相同"
    assert XTrain.shape[1] == x.shape[0], "训练样本数据特性个数应该与被预测数据特性个数相同"

    distances = [sqrt(np.sum((xTrain - x) ** 2)) for xTrain in XTrain]
    nearest = np.argsort(distances)

    topKy = [yTrain[i] for i in nearest[:k]]
    votes = Counter(topKy)

    return votes.most_common(1)[0][0]
