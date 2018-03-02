import numpy as np

def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], \
        "y_true 和 y_predict 数据的行数必须一致"

    return sum(y_true == y_predict) / len(y_predict)