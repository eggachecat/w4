from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import w4_data_processing
import w4_evalution
import w4_extract_feature
import numpy as np
import w4_merge
import json
from scipy.sparse import vstack
import numpy

import time
import pandas as pd

from sklearn.metrics import make_scorer

from sklearn.externals import joblib


def LogLossSearch(clf, X, y):
    y_pred = y, clf.predict_proba(X)

    # the larger the better
    return -w4_evalution.get_logloss(y, y_pred)


def train_once(adv_id, n_max_depth, X, y):
    clf = GridSearchCV(estimator=RandomForestClassifier(n_estimators=500, max_depth=n_max_depth, max_features=None),
                       cv=10)
    loss = clf.fit(X, y)

    return clf, loss


class ConstantClassifier:
    def __init__(self):
        pass

    def predict_proba(self, X):
        return [[1 - self.pred, self.pred] for _ in range(X.shape[0])]

    def fit(self, X, y):
        self.pred = np.sum(y > 0) / len(y)


def train(adv_id):
    _X, _y = w4_data_processing.read_train_tabular(adv_id, low_dim=False)

    train_X, valid_X, train_y, valid_y = train_test_split(_X, _y, test_size=0.3)

    b_clf = ConstantClassifier()
    b_clf.fit(valid_X, valid_y)

    best_loss = benchmark = w4_evalution.get_logloss(valid_y, b_clf.predict_proba(valid_X))

    min_loss_dist = 1000
    test_X, test_y = w4_data_processing.read_train_tabular(adv_id, low_dim=False, w=True)

    for n_max_depth in range(1, 8):

        clf = RandomForestClassifier(n_estimators=1000, max_features=None, max_depth=n_max_depth, n_jobs=-1)
        clf.fit(train_X, train_y)

        in_loss = w4_evalution.get_logloss(train_y, clf.predict_proba(train_X))
        valid_loss = w4_evalution.get_logloss(valid_y, clf.predict_proba(valid_X))

        loss_dist = valid_loss - in_loss

        if loss_dist < min_loss_dist:
            min_loss_dist = loss_dist

            if valid_loss < best_loss:
                best_loss = valid_loss
                b_clf = clf
                b_clf.fit(_X, _y)

        else:
            # early stop
            break

    if best_loss == benchmark:
        print(adv_id, "not improved")
    else:
        joblib.dump(b_clf, 'models/model_tree/{f}.pkl'.format(f=adv_id))
        print(adv_id, "in-excess", benchmark - best_loss)

    return b_clf


def predict_by_adv_id(adv_id, store_obj):
    logger_clf = train(adv_id)

    test_X, l_arr = w4_data_processing.read_train_tabular(adv_id, low_dim=False, is_train=False)

    y_pred = logger_clf.predict_proba(test_X)

    for i in range(len(l_arr)):
        store_obj[str(l_arr[i])] = y_pred[i][1]

    return store_obj


if __name__ == "__main__":
    w4_merge.merge_predict_parallel("split_baseline_rfs.csv", predict_by_adv_id)
