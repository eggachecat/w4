import CorEx.corex as ce
import json
import w4_data_processing
import numpy as np
import sklearn.utils


# def use_corex_on_adv():
#     """
#     can only
#     :return:
#     """
#     with open("data/cache/adv_id_in_test_data.json", "r") as fp:
#         adv_id_arr = json.loads(fp.read())
#
#     with open("data/cache/ce.txt", "w") as fp:
#         for adv_id in adv_id_arr:
#             X, y = w4_data_processing.read_train_tabular(adv_id, feature_filter=[0], preserve=False)
#
#             layer = ce.Corex(n_hidden=50)
#             layer.fit(X.toarray())
#
#             print(layer.clusters)
#             print(layer.tcs)
#
#             fp.write(
#                 "{id}; {cluster}; {tcs} \n".format(id=adv_id, cluster=layer.clusters.rstrip(), tcs=layer.tcs.rstrip()))

def use_corex_on_adv(adv_id, n_hidden):
    """
    :return: 
    """

    features_arr, y = w4_data_processing.read_train_tabular(adv_id, feature_filter=None, preserve=False)
    features_arr = features_arr.toarray()
    # print(features_arr.shape)
    # n_data = len(features_arr)
    # print(n_data)

    # for i in range(n_data):
    #     features_arr[i] = set(features_arr[i])

    features_arr = np.vstack({tuple(row) for row in features_arr})

    # n_data = len(features_arr)
    # print(n_data)
    # print(features_arr)

    # X = w4_data_processing.convert_to_sparse(features_arr)
    # X = sklearn.utils.resample(X, n_samples=n_samples, random_state=0, replace=False)

    layer = ce.Corex(n_hidden=n_hidden)
    layer.fit(features_arr)

    return layer.clusters.tolist()
    # return "{n_hidden}; {clusters}; {tcs} \n".format(n_hidden=n_hidden, clusters=np.array_str(layer.clusters).strip(),
    #                                                  tcs=np.array_str(layer.tcs).strip())


def filter(adv_id_arr):
    n_data = len(adv_id_arr)
    indices_arr = []

    for i in range(n_data):
        if not adv_id_arr[i] == [0]:
            indices_arr.append(i)

    return indices_arr


def use_corex_on_raw(file_path, n_hidden, n_samples=10000):
    """
    :return: 
    """
    adv_id_arr, ts_arr, features_arr = w4_data_processing.read_data(file_path, verbose=True,
                                                                    is_train_format=False)

    n_data = len(features_arr)
    print(n_data)

    # for i in range(n_data):
    #     features_arr[i] = set(features_arr[i])

    features_arr = set(frozenset(i) for i in features_arr)
    features_arr = [list(item) for item in features_arr]

    n_data = len(features_arr)
    print(n_data)

    X = w4_data_processing.convert_to_sparse(features_arr)
    X = sklearn.utils.resample(X, n_samples=n_samples, random_state=0, replace=False)

    layer = ce.Corex(n_hidden=n_hidden)
    layer.fit(X.toarray())
    return "{n_hidden}; {clusters}; {tcs} \n".format(n_hidden=n_hidden, clusters=np.array_str(layer.clusters).strip(),
                                                     tcs=np.array_str(layer.tcs).strip())


if __name__ == "__main__":
    # print(np.__version__)
    # t = [[1, 2], [1, 2], [1, 2], [5], [1, 2, 5], [1, 5, 2], [1, 2, 3, 4], [1, 2, 3, 6]]
    # t1 = set(frozenset(i) for i in t)
    # print(t1)
    # t1 = [list(item) for item in t1]
    # print(t1)
    pass
    # record = use_corex_on_adv("id-613241", n_hidden=20)
    # print(type(record.tolist()))
    # train_paths, test_paths = w4_data_processing.get_origin_data_path()
    #
    # record = use_corex_on_raw(test_paths, 136)
    # print(record)
    #
    # record = use_corex_on_raw(test_paths, 136, n_samples=20000)
    # print(record)
    #
    # record = use_corex_on_raw(test_paths, 136, n_samples=20000)
    # print(record)
    #
    # record = use_corex_on_raw(test_paths, 136, n_samples=40000)
    # print(record)




    # for file_path in reversed(train_paths):
    #     print(file_path)
    #     record = use_corex_on_raw(file_path, 100)
    #     print(record)
    #     fp.write(record)
    #
    # for file_path in test_paths:
    #     print(file_path)
    #     record = use_corex_on_raw(file_path, 100)
    #     print(record)
    #     fp.write(record)
    # map_str = "9 2 2 2 2 2 2 2 2 2 2 2 3 3 1 1 0 2 0 2 0 0 0 \
    #    0 0 0 0 2 0 0 0 0 9 9 0 0 2 8 0 8 9 0 8 7 0 9 2 0 9 2 2 9 0 \
    #    2 0 9 0 2 9 7 0 6 9 0 2 4 2 8 9 6 0 1 6 0 6 5 6 6 4 8 9 8 9 9 1 \
    #    6 6 6 6 5 6 6 8 6 9 9 9 5 5 9 6 6 9 4 3 6 6 6 9 8 5 6 6 6 6 6 6 " \
    #           "6 5 5 5 6 6 5 5 9 2 6 9 5 9 5 6 9 1 8"
    # group = np.fromstring(map_str, dtype=int, sep=' ')
    # print(group)

    # map_str_613241 = "107  28  57  28  26  57  57  10  37  50   6   6   2   2   2   2   1  10\
    #               11  10   0   0   3   1   1   1   3  57  22  28  45  35  42 107  34   1\
    #               10  46  54  39 107  44  51  16  33 107  10 107 107  30  30 107  22  30\
    #              107 107  45  17 107  16  24  29 107  11  47   4  57  34 107  28  26  98\
    #               11  50  26   4  17  11   4  46 107 107 107 107  29  26  26  17  29   4\
    #               17  30  51  17 107 107 107 107  37 107  47  17 107   4  57 107  30 107\
    #              107  47 107 107  30  17  17  37 107 107 107 107   4 107  47 107 107 107\
    #               29 107 107 107 107   4 107 107 107 107"