import w4_data_processing
import w4_evalution
import sklearn.linear_model as sk_lin

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


def tunned_LogReg():
    X, y = w4_data_processing.read_tabular_by_adv_id("id-576772", filter=True)

    folders = w4_data_processing.spilt_data(X, y, 2)

    f1_score_arr = []
    for train_index, test_index in folders:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print(y_train)
        clf = sk_lin.LogisticRegression()
        clf.fit(X_train, y_train)

        prediction = clf.predict(X_test)

        f1_score = w4_evalution.f1_score(y_test, prediction)
        f1_score_arr.append(f1_score)
        print("f1 score is ", f1_score)


# tunned_LogReg()
