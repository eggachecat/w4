import w4_data_processing
import w4_evalution
import sklearn.svm as sk_svm

import numpy as np
from sklearn.model_selection import *
from sklearn.metrics import *


def tunned_SVC(adv_id, low_dim=False):
    X, y = w4_data_processing.read_train_tabular(adv_id, feature_filter=[9], preserve=False, low_dim=low_dim)

    folders = w4_data_processing.spilt_data(X, y, 2)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=0)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                         'C': [0.1, 0.01, 1, 10], 'class_weight': ['balanced']}]

    clf = GridSearchCV(sk_svm.SVC(), tuned_parameters, cv=5,
                       scoring=make_scorer(w4_evalution.f1_score))
    clf.fit(X_train, y_train)

    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    y_true, y_pred = y_valid, clf.predict(X_valid)

    print(y_true, y_pred)

    print(classification_report(y_pred, y_true))

    # f1_score_arr = []
    # for train_index, test_index in folders:
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     print(np.sum(y_train) / len(y_train))
    #
    #     clf = sk_svm.SVC()
    #
    #     clf.fit(X_train, y_train)
    #
    #     prediction = clf.predict(X_test)
    #     print(np.sum(prediction))
    #
    #     f1_score = w4_evalution.get_f1_score(y_test, prediction)
    #     f1_score_arr.append(f1_score)
    #     print("f1 score is ", f1_score)
    #     print("e_in is", w4_evalution.get_error_rate(y_test, prediction))


def tunned_LinearSVC():
    X, y = w4_data_processing.read_train_tabular("id-576772")

    folders = w4_data_processing.spilt_data(X, y, 2)

    f1_score_arr = []
    for train_index, test_index in folders:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(np.sum(y_train) / len(y_train))

        clf = sk_svm.LinearSVC()

        clf.fit(X_train, y_train)

        prediction = clf.predict(X_test)
        print(np.sum(prediction))

        f1_score = w4_evalution.get_f1_score(y_test, prediction)
        f1_score_arr.append(f1_score)
        print("f1 score is ", f1_score)
        print("e_in is", w4_evalution.get_error_rate(prediction, y_test))


if __name__ == "__main__":
    tunned_SVC("id-576772", True)
