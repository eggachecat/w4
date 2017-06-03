import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from scipy.sparse import coo_matrix

import time
import datetime
import json
import os

NUMB_OF_FEATURES = 136
NUMB_OF_ADV = 623
NUMB_OF_TEST = 4859648

TABULAR_ROOT_PATH = "data/tabular"
TABULAR_TS_ROOT_PATH = "data/tabular_ts"
ORIGIN_ROOT_PATH = "data/database"
CACHE_ROOT_PATH = "data/cache"


def get_origin_data_path():
    train_file_paths = []
    for suffix in range(0, 8):
        train_file_paths.append("{org}/train_full_{id}".format(org=ORIGIN_ROOT_PATH, id=suffix))
    for suffix in range(8, 15):
        train_file_paths.append("{org}/train_half_{id}".format(org=ORIGIN_ROOT_PATH, id=suffix))

    test_file_paths = []
    for suffix in range(8, 15):
        test_file_paths.append("{org}/test_half_{id}".format(org=ORIGIN_ROOT_PATH, id=suffix))

    return train_file_paths, test_file_paths


def get_tabular_data_path():
    pass


def timestamp_to_data(timestamp):
    hour, minute = tuple(datetime.datetime.fromtimestamp(int(timestamp)).strftime('%H:%M').split(":"))
    return int(hour), int(minute)


def parse_train_data(line):
    components = str.split(line.rstrip(), " ")
    timestamp = int(components[0])
    adv_id = components[1]
    click = int(components[2])
    feature = [-1 + int(f) for f in components[4:]]

    return timestamp, adv_id, click, feature


def parse_test_data(line):
    components = str.split(line.rstrip(), " ")
    timestamp = int(components[0])
    adv_id = components[1]
    feature = [-1 + int(f) for f in components[3:]]

    return timestamp, adv_id, feature


def read_origin_train_data_ts(file_path_arr):
    """

    :param file_path_arr: [string-arr] or string
    :return adv_id_arr: 
            click_truth_arr
            features_arr
    """
    adv_id_arr = []
    ts_arr = []
    click_truth_arr = []
    features_arr = []

    if not isinstance(file_path_arr, list):
        file_path_arr = [file_path_arr]

    for file_path in file_path_arr:
        with open(file_path, "r") as f:
            for line in f:
                timestamp, adv_id, click, feature = parse_train_data(line)
                ts_arr.append(timestamp)
                adv_id_arr.append(adv_id)
                click_truth_arr.append(click)
                features_arr.append(feature)

    return adv_id_arr, click_truth_arr, ts_arr, features_arr


def read_origin_test_data(file_path_arr):
    """

    :param file_path_arr: [string-arr] or string
    :return adv_id_arr: 
            features_arr:
    """
    adv_id_arr = []
    features_arr = []
    ts_arr = []

    if not isinstance(file_path_arr, list):
        file_path_arr = [file_path_arr]

    for file_path in file_path_arr:
        print(file_path)
        with open(file_path, "r") as f:
            for line in f:
                components = str.split(line.rstrip(), " ")
                ts_arr.append(int(components[0]))
                adv_id_arr.append(components[1])
                features_arr.append([-1 + int(f) for f in components[3:]])

    return adv_id_arr, ts_arr, features_arr


def convert_origin_to_tabular_ts(train_file_paths=None):
    fp = open("{org}/adv_id_in_test_data.json".format(org=CACHE_ROOT_PATH), "r")
    adv_in_test = json.loads(fp.read())

    if train_file_paths is None:
        train_file_paths, predict_file_paths = get_origin_data_path()

    for train_file in train_file_paths:
        adv_recorder = dict()
        for adv_id in adv_in_test:
            adv_recorder[adv_id] = []

        print("Start parsing data file {filename}".format(filename=train_file))
        adv_id_arr, click_truth_arr, ts_arr, features_arr = read_origin_train_data_ts([train_file])
        n_data = len(click_truth_arr)
        for i in range(n_data):
            adv_id = adv_id_arr[i]
            if adv_id in adv_in_test:
                record = "{truth} {ts} {feature}".format(truth=click_truth_arr[i], ts=ts_arr[i],
                                                         feature=features_arr[i])
                adv_recorder[adv_id].append(record)

        for adv_id in adv_in_test:
            with open("{r}/{f}.txt".format(r=TABULAR_TS_ROOT_PATH, f=adv_id), "a+") as fp:
                for record in adv_recorder[adv_id]:
                    fp.write(record + "\n")


def convert_origin_predict_to_tabular_ts():
    cache_path = "data/cache/test_data_group_by_adv.json"
    store_obj = dict()

    if os.path.isfile(cache_path):
        print("loading...")
        start = time.time()
        with open(cache_path, 'r') as fp:
            store_obj = json.loads(fp.read())
        end = time.time()

        print("done {t}s".format(t=end - start))

    else:
        train_file_paths, test_file_paths = get_origin_data_path()
        print("start...")
        adv_id_arr, ts_arr, features_arr = read_origin_test_data(test_file_paths)

        for i in range(len(adv_id_arr)):

            adv_id = adv_id_arr[i]
            if adv_id not in store_obj:
                store_obj[adv_id] = []
            store_obj[adv_id].append({
                "l": i,
                "u": features_arr[i],
                "t": ts_arr[i]
            })
        with open(cache_path, 'w') as fp:
            json.dump(store_obj, fp)

    for adv_id in store_obj:
        root_path = "data/tabular_ts_predict/{id}".format(id=adv_id)
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        with open("{rp}/test.txt".format(rp=root_path), 'w') as fp:
            for record in store_obj[adv_id]:
                fp.write("{l} {t} {u} \n".format(l=record["l"], t=timestamp_to_data(record["t"])[0], u=record["u"]))


# convert_origin_predict_to_tabular_ts()


def read_origin_train_data(file_path_arr):
    """
    
    :param file_path_arr: [string-arr] or string
    :return adv_id_arr: 
            click_truth_arr
            features_arr
    """
    adv_id_arr = []
    click_truth_arr = []
    features_arr = []

    if not isinstance(file_path_arr, list):
        file_path_arr = [file_path_arr]

    for file_path in file_path_arr:
        with open(file_path, "r") as f:
            for line in f:
                components = str.split(line.rstrip(), " ")

                adv_id_arr.append(components[1])
                click_truth_arr.append(int(components[2]))
                features_arr.append([-1 + int(f) for f in components[4:]])

    return adv_id_arr, click_truth_arr, features_arr


def read_tabular_ts_by_adv_id(adv_id, feature_filter=None, preserve=False, with_ts=False):
    """
    
    :param adv_id: 
    :param feature_filter: [list-of-feature]
            used to filter our data
    :param preserve: [bool]
            if preserve is true, then the one have the same feature with filter will be taken into account
    :param with_ts: 
    :return: 
    """
    click_truth_arr = []
    features_arr = []
    ts_arr = []

    file_path = "{r}/{f}.txt".format(r=TABULAR_TS_ROOT_PATH, f=adv_id)
    with open(file_path, "r") as f:
        for line in f:
            components = str.split(line.rstrip(), " ", 2)

            if feature_filter:
                feature_match = eval(components[2]) == feature_filter
                if preserve == feature_match:
                    pass
                else:
                    continue

            click_truth_arr.append(int(components[0]))
            features_arr.append(eval(components[2]))
            if with_ts:
                ts_arr.append(int(components[1]))

    y = click_truth_arr

    # convert features_arr to sparse matrix
    row = [i for i in range(len(features_arr)) for _ in range(len(features_arr[i]))]
    col = [item for sublist in features_arr for item in sublist]
    data = np.ones(len(row), dtype=int)
    X = coo_matrix((data, (row, col)), shape=(len(click_truth_arr), NUMB_OF_FEATURES))

    if with_ts:
        day_trend = [timestamp_to_data(t) for t in ts_arr]
        return X.tocsr(), np.array(y), day_trend
    else:
        return X.tocsr(), np.array(y)


def read_tabular_by_adv_id(adv_id, filter=False):
    """
    
    :param file_path_arr: 
    :param adv_id: 
    :return: 
    """
    click_truth_arr = []
    features_arr = []

    file_path = "{r}/{f}.txt".format(r=TABULAR_ROOT_PATH, f=adv_id)
    with open(file_path, "r") as f:
        for line in f:

            components = str.split(line.rstrip(), " ", 1)
            if filter:
                if eval(components[1]) == [0]:
                    continue
            click_truth_arr.append(int(components[0]))
            features_arr.append(eval(components[1]))

    y = click_truth_arr

    # convert features_arr to sparse matrix
    row = [i for i in range(len(features_arr)) for _ in range(len(features_arr[i]))]
    col = [item for sublist in features_arr for item in sublist]
    data = np.ones(len(row), dtype=int)
    X = coo_matrix((data, (row, col)), shape=(len(click_truth_arr), NUMB_OF_FEATURES))

    # More readable
    # row = []
    # col = []
    # for i in range(len(features_arr)):
    #     for x in features_arr[i]:
    #         row.append(i)
    #         col.append(x)
    # data = np.ones(len(row), dtype=float)
    # X = coo_matrix((data, (row, col)), shape=(len(click_truth_arr), NUMB_OF_FEATURES))

    return X.tocsr(), np.array(y)


def spilt_data(X, y, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(X, y)

    return skf.split(X, y)


def save_to_stack():
    pass

# start = time.time()
# end = time.time()
# print(end - start)

#
# X = [[2], [2, 3], [1]]
# n_row = 50000
# n_col = 100
# X = [np.random.choice(n_col,  np.random.random_integers(n_col), replace=False) for _ in range(n_row)]
#
# start = time.time()
# row = [i for i in range(len(X)) for _ in range(len(X[i]))]
# col = [item for sublist in X for item in sublist]
# data = np.ones(len(row), dtype=int)
# m = coo_matrix((data, (row, col)), shape=(n_row, n_col))
# end = time.time()
# print(end-start)
#
# start = time.time()
#
# row = []
# col = []
# for i in range(len(X)):
#     for x in X[i]:
#         row.append(i)
#         col.append(x)
# data = np.ones(len(row), dtype=int)
# m = coo_matrix((data, (row, col)), shape=(n_row, n_col))
# end = time.time()
# print(end-start)
# features_arr = [[2], [2, 3], [0]]
# y = [0, 0, 1]
#

# X, y = read_tabular_by_adv_id(TABULAR_ROOT_PATH + "id-579985.txt")
# spilt_data(X, y)
