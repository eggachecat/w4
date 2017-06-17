import numpy as np

from sklearn.naive_bayes import BernoulliNB

import w4_evalution

import w4_data_processing
import w4_merge

from sklearn.model_selection import *
from sklearn.metrics import *
import json
import w4_extract_feature


def train(adv_id, verbose=False, n_min=2, n_max=11):
    best_map = None
    best_clf = None
    min_logloss = 100

    for n_hidden in range(n_min, n_max):
        group_map = w4_extract_feature.use_corex_on_adv(adv_id, n_hidden=n_hidden)
        X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=group_map)

        clf = BernoulliNB()
        clf.fit(X, y)
        y_true, y_pred = y, clf.predict_proba(X)
        log_loss = w4_evalution.get_logloss(y_true, y_pred)

        if verbose:
            print(n_hidden, log_loss)
        if log_loss < min_logloss:
            min_logloss = log_loss
            best_clf = clf
            best_map = group_map

    return best_clf, best_map


# tune("id-591573", low_dim=True)

# def predict_by_adv_id(adv_id, store_obj, low_dim=False, group_map=None):
#     src_root_path = "data/tabular_ts_test_ld"
#
#     l_arr = []
#     feature_arr = []
#
#     with open("{rp}/{id}.txt".format(rp=src_root_path, id=adv_id), 'r') as f:
#         for line in f:
#             components = str.split(line.rstrip(), " ", 2)
#             l = int(components[0])
#             feature = eval(components[2])
#             l_arr.append(l)
#             feature_arr.append(feature)
#
#     X, l_arr = w4_data_processing.read_train_tabular(adv_id, low_dim=low_dim, group_map=group_map)
#
#     X = w4_data_processing.convert_to_sparse(feature_arr)
#
#     clf = train(adv_id)
#     pred = clf.predict_proba(X)
#
#     for i in range(len(l_arr)):
#         store_obj[l_arr[i]] = pred[i][1]
#
#     return store_obj

def predict_by_adv_id(adv_id, store_obj):
    clf, group_map = train(adv_id)

    X, l_arr = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=group_map, is_train=False)
    pred = clf.predict_proba(X)

    X_, y = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=None, w=True)
    print(len(y), w4_evalution.get_logloss(y, pred))

    for i in range(len(l_arr)):
        store_obj[str(l_arr[i])] = pred[i][1]

    return store_obj


def super_naive_predict(adv_id, store_obj):
    X_train, y_train = w4_data_processing.read_train_tabular(adv_id, low_dim=False)
    pred = np.sum(y_train > 0) / len(y_train)

    X, l_arr = w4_data_processing.read_train_tabular(adv_id, low_dim=False, is_train=False)

    for i in range(len(l_arr)):
        store_obj[str(l_arr[i])] = pred

    return store_obj


if __name__ == "__main__":
    w4_merge.merge_predict_parallel("super_naive.csv", super_naive_predict)

    pass
# adv_id = "id-611482"
#
# X_train, y_train = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=None, w=False)
# print(np.sum(y_train > 0) / len(y_train))
# x = np.sum(y_train > 0) / len(y_train)
#
# X_test, y_test = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=None, w=True)
# y_pred = x * np.ones(y_test.shape, dtype=float)
# x = np.sum(y_test > 0) / len(y_test)
#
# print("out", w4_evalution.get_logloss(y_test, y_pred))



# train("id-595770", verbose=True, n_min=2, n_max=50)

# w4_merge.merge_predict("shit_bayes.csv", predict_by_adv_id)
# adv_id = "id-591728"
# clf = train(adv_id, low_dim=True, group_map=None)
# X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=True, group_map=None, w=True)
# y_true, y_pred = y, clf.predict_proba(X)
# print(w4_evalution.get_logloss(y_true, y_pred))

# for n_hidden in range(2, 11):
#     group_map = w4_extract_feature.use_corex_on_adv(adv_id, n_hidden=n_hidden)
#     clf = train(adv_id, low_dim=False, group_map=group_map)
#
#     X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=group_map, w=True)
#     y_true, y_pred = y, clf.predict_proba(X)
#     print("out", w4_evalution.get_logloss(y_true, y_pred))
#     print("----------")

# with open("data/cache/adv_id_in_test_data.json", "r") as fp:



#     adv_id_arr = json.loads(fp.read())
# w4_merge.merge_predict("outputs/improved_bayes.csv", predict_by_adv_id)
# adv_id = "id-613241"
#
# with open("data/cache/adv_id_in_test_data.json", "r") as fp:
#     adv_id_arr = json.loads(fp.read())
#
# for adv_id in adv_id_arr:
#     print(adv_id)
#     clf = train(adv_id, low_dim=True, group_map=None)
#     X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=True, group_map=None, w=True)
#     y_true, y_pred = y, clf.predict_proba(X)
#     print(w4_evalution.get_logloss(y_true, y_pred))
#
#     n_hidden = 2
#     group_map = w4_extract_feature.use_corex_on_adv(adv_id, n_hidden=n_hidden)
#     clf = train(adv_id, low_dim=False, group_map=group_map)
#
#     X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=group_map, w=True)
#     y_true, y_pred = y, clf.predict_proba(X)
#     print(w4_evalution.get_logloss(y_true, y_pred))
#     print("----------")




# X_data, l_arr = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=group, is_train=False)
# X, y = w4_data_processing.read_train_tabular(adv_id, group_map=None, low_dim=True, w=True)

# y_true, y_pred = y, clf.predict_proba(X)
# print(adv_id, w4_evalution.get_logloss(y_true, y_pred))

# X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=True, w=True)
# y_true, y_pred = y, clf.predict_proba(X)
# print(adv_id, w4_evalution.get_logloss(y_true, y_pred))

# with open("data/cache/adv_id_in_test_data.json", "r") as fp:
#     adv_id_arr = json.loads(fp.read())
#
# logloss_arr = []
# for adv_id in adv_id_arr:
#     clf = train(adv_id, low_dim=True)
#     X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=True, w=True)
#     y_true, y_pred = y, clf.predict_proba(X)
#     print(adv_id, w4_evalution.get_logloss(y_true, y_pred))
#     logloss_arr.append(w4_evalution.get_logloss(y_true, y_pred))
#
# print(np.mean(logloss_arr))
# print(np.std(logloss_arr))



# w4_merge.merge_predict("outputs/bayes.csv", predict_by_adv_id)
