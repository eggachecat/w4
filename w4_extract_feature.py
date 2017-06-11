import CorEx.corex as ce
import json
import w4_data_processing
import numpy as np
import sklearn.utils


def use_corex_on_adv():
    """
    can only
    :return: 
    """
    with open("data/cache/adv_id_in_test_data.json", "r") as fp:
        adv_id_arr = json.loads(fp.read())

    with open("data/cache/ce.txt", "w") as fp:
        for adv_id in adv_id_arr:
            X, y = w4_data_processing.read_train_tabular(adv_id, feature_filter=[0], preserve=False)

            layer = ce.Corex(n_hidden=50)
            layer.fit(X.toarray())

            print(layer.clusters)
            print(layer.tcs)

            fp.write(
                "{id}; {cluster}; {tcs} \n".format(id=adv_id, cluster=layer.clusters.rstrip(), tcs=layer.tcs.rstrip()))


def filter(adv_id_arr):
    n_data = len(adv_id_arr)
    indices_arr = []

    for i in range(n_data):
        if not adv_id_arr[i] == [0]:
            indices_arr.append(i)

    return indices_arr


def use_corex_on_raw(file_path, n_hidden):
    """
    :return: 
    """
    adv_id_arr, ts_arr, features_arr = w4_data_processing.read_data(file_path, verbose=True,
                                                                    is_train_format=False)

    X = w4_data_processing.convert_to_sparse(features_arr)
    filtered_indices = filter(adv_id_arr)
    X = X[filtered_indices]
    X = sklearn.utils.resample(X, n_samples=20000, random_state=0)

    layer = ce.Corex(n_hidden=n_hidden)
    layer.fit(X.toarray())
    return "{n_hidden}; {clusters}; {tcs} \n".format(n_hidden=n_hidden, clusters=np.array_str(layer.clusters).strip(),
                                                     tcs=np.array_str(layer.tcs).strip())


# if __name__ == "__main__":
#     train_paths, test_paths = w4_data_processing.get_origin_data_path()
#
#     with open("data/cache/all_raw_ce.txt", "w") as fp:
#         record = use_corex_on_raw(test_paths, 10)
#         print(record)
#         record = use_corex_on_raw(test_paths, 20)
#         print(record)
#         record = use_corex_on_raw(test_paths, 50)
#         print(record)
#         fp.write(record)
#         record = use_corex_on_raw(test_paths, 75)
#         print(record)
#         record = use_corex_on_raw(test_paths, 100)
#         print(record)
#         fp.write(record)

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
map_str = "9 2 2 2 2 2 2 2 2 2 2 2 3 3 1 1 0 2 0 2 0 0 0 \
   0 0 0 0 2 0 0 0 0 9 9 0 0 2 8 0 8 9 0 8 7 0 9 2 0 9 2 2 9 0 \
   2 0 9 0 2 9 7 0 6 9 0 2 4 2 8 9 6 0 1 6 0 6 5 6 6 4 8 9 8 9 9 1 \
   6 6 6 6 5 6 6 8 6 9 9 9 5 5 9 6 6 9 4 3 6 6 6 9 8 5 6 6 6 6 6 6 " \
          "6 5 5 5 6 6 5 5 9 2 6 9 5 9 5 6 9 1 8"
group = np.fromstring(map_str, dtype=int, sep=' ')
print(group)