from sklearn.metrics import *
import numpy as np
import w4_data_processing
import json


def get_f1_score(truth, prediction):
    return f1_score(truth, prediction)


def get_error_rate(truth, prediction):
    if not len(truth) == len(prediction):
        print("truth and prediction should have the same length")
        exit()

    return 1 - np.sum(truth == prediction) / len(truth)


def get_logloss(truth, prob):
    return log_loss(truth, prob)


def get_benchmark(adv_id):
    X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=False)
    pred = np.sum(y > 0) / len(y)

    return -pred * np.log(pred) - (1 - pred) * np.log(1 - pred)


def get_out_benchmark(adv_id):
    X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=False)
    pred = np.sum(y > 0) / len(y)

    X_test, y_test = w4_data_processing.read_train_tabular(adv_id, low_dim=False, w=True)

    return get_logloss(y_test, pred * np.ones(y_test.shape))


if __name__ == "__main__":
    with open("data/cache/adv_id_in_test_data.json", "r") as fp:
        adv_obj_arr = json.loads(fp.read())

    adv_obj_arr = list(reversed(sorted(adv_obj_arr.items(), key=lambda x: x[1])))
    in_b_arr = []
    o_b_arr = []
    for adv_obj in adv_obj_arr:
        adv_id = adv_obj[0]
        in_b = get_benchmark(adv_id)
        o_b = get_out_benchmark(adv_id)
        in_b_arr.append(in_b)
        o_b_arr.append(o_b)
        print(adv_id, in_b_arr[-1], o_b_arr[-1])

    in_b_arr = np.array(in_b_arr)
    o_b_arr = np.array(o_b_arr)

    df = in_b_arr - o_b_arr
    print(np.mean(df), np.std(df))
