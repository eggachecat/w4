import w4_data_processing
import w4_evalution
import sklearn.linear_model as sk_lin

import numpy as np
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.metrics import classification_report

from sklearn.ensemble import GradientBoostingClassifier


class W4LogisticRegression(sk_lin.LogisticRegression):
    def predict(self, X):
        return super(W4LogisticRegression, self).predict_log_proba(X)

    def _get_param_names(self):
        return super(W4LogisticRegression, self)._get_param_names()


class W4LogReg:
    def __init__(self):
        pass

    def train(self, adv_id):
        X, y = w4_data_processing.read_train_tabular(adv_id, feature_filter=None, preserve=False)

        print(len(y), np.sum(y))
        folders = w4_data_processing.spilt_data(X, y, 2)

        f1_score_arr = []
        for train_index, test_index in folders:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = sk_lin.LogisticRegression(class_weight="balanced")
            clf.fit(X_train, y_train)

            prediction = clf.predict(X_test)

            f1_score = w4_evalution.get_f1_score(y_test, prediction)
            f1_score_arr.append(f1_score)
            print("f1 score is ", f1_score)

            err_rate = w4_evalution.get_error_rate(y_test, prediction)
            print("err_rate is ", err_rate)

    def tune(self, adv_id, low_dim=False):
        # in low_dim 0 ->9

        X, y = w4_data_processing.read_train_tabular(adv_id, feature_filter=[9], preserve=False, low_dim=low_dim)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.1, random_state=0)

        tuned_parameters = [
            {'class_weight': [None, "balanced"],
             'tol': [10, 1, 1e-1, 1e-2], "C": [1, 1e-1, 1e-2, 1e-3],
             "penalty": ['l1', 'l2']}]

        print("# Tuning hyper-parameters for %s" % "f1")
        print()

        clf = GridSearchCV(W4LogisticRegression(), tuned_parameters, cv=5,
                           scoring=make_scorer(w4_evalution.get_logloss))
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        # print()

        # print("Detailed classification report:")
        # print()
        # print("The model is trained on the full development set.")
        # print("The scores are computed on the full evaluation set.")
        # print("hi!!!!")
        y_true, y_pred = y_valid, clf.predict(X_valid)

        print(w4_evalution.get_logloss(y_true, y_pred))

        # print(np.sum(y_true > 0))
        # print(np.sum(y_pred > 0))
        # print(y_true, y_pred)
        # print(y_pred)
        # print(np.exp(clf.predict_log_proba(X_valid)))
        #
        # print(classification_report(y_true, y_pred))
        # print()


if __name__ == "__main__":
    model = W4LogReg()
    model.tune("id-595021", True)
    # from sklearn.metrics import classification_report
    #
    # y_true = [0, 1, 2, 2, 2]
    # y_pred = [0, 0, 2, 2, 1]
    # target_names = ['class 0', 'class 1', 'class 2']
    # print(classification_report(y_true, y_pred, target_names=target_names))

# tunned_LogReg()
