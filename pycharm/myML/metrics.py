import numpy as np

def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], \
        "y_true 和 y_predict 数据的行数必须一致"

    return np.sum(y_true == y_predict) / len(y_predict)

def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), "y_true与y_predict的数量必须一致"

    return np.sum((y_true - y_predict)**2) / len(y_true)

def root_mean_squared_error(y_true, y_predict):
    return np.sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), "y_true与y_predict的数量必须一致"

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true, y_predict):
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)