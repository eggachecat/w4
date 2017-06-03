import w4_data_processing
import w4_evalution
import sklearn.svm as sk_svm

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


def tunned_SVC():
    X, y = w4_data_processing.read_tabular_by_adv_id("id-576772", filter=True)

    folders = w4_data_processing.spilt_data(X, y, 2)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    f1_score_arr = []
    for train_index, test_index in folders:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(np.sum(y_train) / len(y_train))

        clf = sk_svm.SVC()

        clf.fit(X_train, y_train)

        prediction = clf.predict(X_test)
        print(np.sum(prediction))

        f1_score = w4_evalution.f1_score(y_test, prediction)
        f1_score_arr.append(f1_score)
        print("f1 score is ", f1_score)
        print("e_in is", w4_evalution.get_error_rate(y_test, prediction))


def tunned_LinearSVC():
    X, y = w4_data_processing.read_tabular_by_adv_id("id-576772", filter=True)

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

        f1_score = w4_evalution.f1_score(y_test, prediction)
        f1_score_arr.append(f1_score)
        print("f1 score is ", f1_score)
        print("e_in is", w4_evalution.get_error_rate(y_test, prediction))


# tunned_LinearSVC()
