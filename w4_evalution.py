from sklearn.metrics import f1_score
import numpy as np


def get_f1_score(truth, prediction):
    return f1_score(truth, prediction)


def get_error_rate(truth, prediction):
    if not len(truth) == len(prediction):
        print("truth and prediction should have the same length")
        exit()

    return np.sum(truth == prediction) / len(truth)

