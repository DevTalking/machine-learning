import numpy as np

# 训练样本数据 X 和 y 按照 test_radio 分割成 X_train, y_train, X_test, y_test
def train_test_split(X, y, test_radio = 0.2, seed = None):
    assert X.shape[0] == y.shape[0], \
        "训练样本特征数据集的行数要与训练样本分类结果数据集的行数相同"
    assert 0.0 <= test_radio <= 1.0, \
        "test_radio 的值必须在 0 到 1 之间"

    # 如果 seed 有值，将其设置进numpy的随机函数中
    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_radio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, y_train, X_test, y_test