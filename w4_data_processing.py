import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from scipy.sparse import coo_matrix
import pandas as pd

import time
import datetime
import json
import os

NUM_OF_FEATURES = 136
NUM_OF_ADV = 623
NUM_OF_TEST = 4859648
NUM_OF_LD_FEATURES = 10

TABULAR_ROOT_PATH = "data/tabular_ts"
TABULAR_LD_ROOT_PATH = "data/tabular_low_dim"

ORIGIN_ROOT_PATH = "data/database"
ORIGIN_LD_ROOT_PATH = "data/database_low_dim"

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


def get_low_dim_origin_data_path():
    train_file_paths = []
    for suffix in range(0, 8):
        train_file_paths.append("{org}/train_full_{id}".format(org=ORIGIN_LD_ROOT_PATH, id=suffix))
    for suffix in range(8, 15):
        train_file_paths.append("{org}/train_half_{id}".format(org=ORIGIN_LD_ROOT_PATH, id=suffix))

    test_file_paths = []
    for suffix in range(8, 15):
        test_file_paths.append("{org}/test_half_{id}".format(org=ORIGIN_LD_ROOT_PATH, id=suffix))

    return train_file_paths, test_file_paths


def timestamp_to_hour(timestamp):
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


def read_data(file_path_arr, is_train_format=True, verbose=False):
    adv_id_arr = []
    ts_arr = []
    click_truth_arr = []
    features_arr = []

    if not isinstance(file_path_arr, list):
        file_path_arr = [file_path_arr]

    for file_path in file_path_arr:
        if verbose:
            print("Start loading", file_path)
        with open(file_path, "r") as f:
            for line in f:
                if is_train_format:
                    timestamp, adv_id, click, feature = parse_train_data(line)
                    click_truth_arr.append(click)
                else:
                    timestamp, adv_id, feature = parse_test_data(line)
                ts_arr.append(timestamp)
                adv_id_arr.append(adv_id)
                features_arr.append(feature)

    if is_train_format:
        return adv_id_arr, click_truth_arr, ts_arr, features_arr
    else:
        return adv_id_arr, ts_arr, features_arr


def read_origin_train_data(file_path_arr):
    if not isinstance(file_path_arr, list):
        file_path_arr = [file_path_arr]

    return read_data(file_path_arr)


def read_origin_test_data(file_path_arr):
    if not isinstance(file_path_arr, list):
        file_path_arr = [file_path_arr]

    return read_data(file_path_arr, is_train_format=False)


def convert_origin_train_to_tabular(get_path_func=get_origin_data_path, root_path=TABULAR_ROOT_PATH):
    fp = open("{org}/adv_id_in_test_data.json".format(org=CACHE_ROOT_PATH), "r")
    adv_in_test = json.loads(fp.read())

    train_file_paths, predict_file_paths = get_path_func()

    for file_path in train_file_paths:
        adv_recorder = dict()
        for adv_id in adv_in_test:
            adv_recorder[adv_id] = []

        print("Start parsing data file {filename}".format(filename=file_path))
        adv_id_arr, click_truth_arr, ts_arr, features_arr = read_origin_train_data([file_path])
        n_data = len(click_truth_arr)
        for i in range(n_data):
            adv_id = adv_id_arr[i]
            if adv_id in adv_in_test:
                record = "{truth} {ts} {feature}".format(truth=click_truth_arr[i], ts=ts_arr[i],
                                                         feature=features_arr[i])
                adv_recorder[adv_id].append(record)

        for adv_id in adv_in_test:

            filename = "{r}/{f}.txt".format(r=root_path, f=adv_id)
            if os.path.exists(filename):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'  # make a new file if not

            with open(filename, append_write) as fp:
                for record in adv_recorder[adv_id]:
                    fp.write(record + "\n")


def convert_origin_train_to_low_dim(group_map):
    """
    Just make sure it matches the original type
    :param group_map: 
    :return: 
    """
    train_file_paths, test_file_paths = get_origin_data_path()

    for train_file in train_file_paths:
        print(train_file)
        adv_id_arr, click_truth_arr, ts_arr, features_arr = read_data(train_file, is_train_format=True)

        file_name = train_file.split("/")[-1]

        n_data = len(features_arr)
        for i in range(n_data):
            features_arr[i] = list(set([str(1 + int(group_map[f])) for f in features_arr[i]]))

        with open("data/{rp}/{f}".format(rp=ORIGIN_LD_ROOT_PATH, f=file_name), "w") as fp:
            for i in range(n_data):
                fp.write("{ts} {id} {c} |user {f}\n".format(ts=ts_arr[i], id=adv_id_arr[i], c=click_truth_arr[i],
                                                            f=" ".join(features_arr[i])))

    for test_file in test_file_paths:
        print(test_file)

        adv_id_arr, ts_arr, features_arr = read_data(test_file, is_train_format=False)
        file_name = test_file.split("/")[-1]

        n_data = len(features_arr)
        for i in range(n_data):
            features_arr[i] = list(set([str(1 + int(group_map[f])) for f in features_arr[i]]))

        with open("data/{rp}/{f}".format(rp=ORIGIN_LD_ROOT_PATH, f=file_name), "w") as fp:
            for i in range(n_data):
                fp.write("{ts} {id} |user {f}\n".format(ts=ts_arr[i], id=adv_id_arr[i], f=" ".join(features_arr[i])))


def convert_origin_test_to_json(cache_path="data/cache/test_data_group_by_adv.json", root_path="data/tabular_ts_test",
                                low_dim=False, test_file_paths=None):
    store_obj = dict()

    if os.path.isfile(cache_path):
        print("loading...")
        start = time.time()
        with open(cache_path, 'r') as fp:
            store_obj = json.loads(fp.read())
        end = time.time()

        print("done {t}s".format(t=end - start))

    else:
        if test_file_paths is None:
            if not low_dim:
                train_file_paths, test_file_paths = get_origin_data_path()
            else:
                train_file_paths, test_file_paths = get_low_dim_origin_data_path()
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
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        with open("{rp}/{id}.txt".format(rp=root_path, id=adv_id), 'w') as fp:
            for record in store_obj[adv_id]:
                fp.write("{l} {t} {u} \n".format(l=record["l"], t=record["t"]
                                                 , u=record["u"]))


def convert_to_sparse(features_arr):
    row = [i for i in range(len(features_arr)) for _ in range(len(features_arr[i]))]
    col = [item for sublist in features_arr for item in sublist]
    data = np.ones(len(row), dtype=int)

    return coo_matrix((data, (row, col)), shape=(len(features_arr), NUM_OF_FEATURES)).tocsr()


def read_train_tabular(adv_id, feature_filter=None, preserve=False, with_ts=False, low_dim=False, w=False,
                       group_map=None, is_train=True):
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

    if is_train:
        if low_dim:
            if w:
                file_path = "{r}/{f}.txt".format(r="data/www_tabular_ld", f=adv_id)
            else:
                file_path = "{r}/{f}.txt".format(r=TABULAR_LD_ROOT_PATH, f=adv_id)
        else:
            if w:
                file_path = "{r}/{f}.txt".format(r="data/www_tabular", f=adv_id)
            else:
                file_path = "{r}/{f}.txt".format(r=TABULAR_ROOT_PATH, f=adv_id)
    else:
        file_path = "data/tabular_ts_test/{f}.txt".format(f=adv_id)

    # print(file_path)
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
            if group_map is None:
                features_arr.append(eval(components[2]))
            else:
                feature = list(set([group_map[f] for f in eval(components[2])]))

                features_arr.append(feature)
            if with_ts:
                ts_arr.append(int(components[1]))

    y = click_truth_arr

    # convert features_arr to sparse matrix
    # row = [i for i in range(len(features_arr)) for _ in range(len(features_arr[i]))]
    # col = [item for sublist in features_arr for item in sublist]
    # data = np.ones(len(row), dtype=int)
    # X = coo_matrix((data, (row, col)), shape=(len(click_truth_arr), NUM_OF_FEATURES))
    # print(features_arr)
    X = convert_to_sparse(features_arr)

    if with_ts:
        day_trend = [timestamp_to_hour(t) for t in ts_arr]
        return X, np.array(y), day_trend
    else:
        return X, np.array(y)


def spilt_data(X, y, n_splits=3):
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(X, y)

    return skf.split(X, y)


def save_to_stack():
    pass


def parse_data_w(line):
    components = str.split(line.rstrip(), " ", maxsplit=4)
    timestamp = components[0]
    adv_id = components[1]
    click = int(components[2])
    feature = str.split(components[4], "|", maxsplit=2)[0]
    feature = str.split(feature.rstrip(), " ")
    feature = [-1 + int(x) for x in feature]

    return timestamp, adv_id, click, feature


def read_data_w(file_path):
    adv_id_arr = []
    ts_arr = []
    click_truth_arr = []
    features_arr = []

    with open(file_path, "r") as f:
        for line in f:
            timestamp, adv_id, click, feature = parse_data_w(line)
            click_truth_arr.append(click)
            ts_arr.append(timestamp)
            adv_id_arr.append(adv_id)
            features_arr.append(feature)

    return adv_id_arr, click_truth_arr, ts_arr, features_arr


def get_path_w():
    return ["data/www_database/ans_{i}.csv".format(i=x) for x in range(8, 15)], []


def convert_origin_train_to_low_dim_w(group_map):
    """
    Just make sure it matches the original type
    :param group_map: 
    :return: 
    """

    test_file_paths = ["data/www/w{i}".format(i=x) for x in range(7)]

    for test_file in test_file_paths:
        print(test_file)

        adv_id_arr, click_truth_arr, ts_arr, features_arr = read_data_w(test_file)
        file_name = test_file.split("/")[-1]

        n_data = len(features_arr)
        for i in range(n_data):
            features_arr[i] = list(set([str(1 + int(group_map[f])) for f in features_arr[i]]))

        with open("data/{rp}/{f}".format(rp="www_low_dim", f=file_name), "w") as fp:
            for i in range(n_data):
                fp.write("{ts} {id} {c} |user {f}\n".format(ts=ts_arr[i], id=adv_id_arr[i], c=click_truth_arr[i],
                                                            f=" ".join(features_arr[i])))


def pd_parse(x):
    return parse_train_data(x[0])


if __name__ == "__main__":
    convert_origin_train_to_tabular(get_path_w, root_path="data/www_tabular")

    # df = pd.read_csv("data/database/train_full_0", header=None, dtype="str")
    #
    # df.apply(pd_parse, axis=1)
    # pass
    # convert_origin_train_to_tabular(get_path_w, "data/www_tabular_ld")
    # pass
    # map_str = "9 2 2 2 2 2 2 2 2 2 2 2 3 3 1 1 0 2 0 2 0 0 0 \
    # 0 0 0 0 2 0 0 0 0 9 9 0 0 2 8 0 8 9 0 8 7 0 9 2 0 9 2 2 9 0 \
    # 2 0 9 0 2 9 7 0 6 9 0 2 4 2 8 9 6 0 1 6 0 6 5 6 6 4 8 9 8 9 9 1 \
    # 6 6 6 6 5 6 6 8 6 9 9 9 5 5 9 6 6 9 4 3 6 6 6 9 8 5 6 6 6 6 6 6 " \
    #           "6 5 5 5 6 6 5 5 9 2 6 9 5 9 5 6 9 1 8"
    # group = np.fromstring(map_str, dtype=int, sep=' ')
    # convert_origin_train_to_low_dim_w(group)
    # # convert_origin_train_to_tabular(get_low_dim_origin_data_path)
    # convert_origin_test_to_json(low_dim=True, cache_path="data/cache/test_data_group_by_adv_ld.json")
