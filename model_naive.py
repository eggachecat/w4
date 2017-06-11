import numpy as np
import w4_data_processing
import json
import time
from pandas import *
import _thread
import threading


def naive_count(low_dim=False):
    with open("data/cache/adv_id_in_test_data.json", "r") as fp:
        adv_id_in_test_arr = json.loads(fp.read())

    clicks = dict()
    selections = dict()

    if low_dim:
        train_paths, test_paths = w4_data_processing.get_low_dim_origin_data_path()
    else:
        train_paths, test_paths = w4_data_processing.get_origin_data_path()

    adv_id_arr, click_truth_arr, ts_arr, features_arr = w4_data_processing.read_data(train_paths, verbose=True)

    n_data = len(click_truth_arr)

    for i in range(n_data):
        adv_id = adv_id_arr[i]
        if adv_id not in adv_id_in_test_arr:
            continue

        click_truth = int(click_truth_arr[i])
        if adv_id not in clicks:
            clicks[adv_id] = dict()
            selections[adv_id] = dict()
            for k in range(w4_data_processing.NUM_OF_FEATURES):
                clicks[adv_id][k] = 0
                selections[adv_id][k] = 1
        for f in features_arr[i]:
            clicks[adv_id][f] += click_truth
            selections[adv_id][f] += 1

    with open("data/cache/clk_stat_for_fea_and_adv_ld.json", 'w') as fp:
        json.dump(clicks, fp)

    with open("data/cache/sel_stat_for_fea_and_adv_ld.json", 'w') as fp:
        json.dump(selections, fp)


def get_adv_prob(clicks, selections):
    sum_c = pandas.DataFrame.sum(clicks, axis=1)
    sum_s = pandas.DataFrame.sum(selections, axis=1)
    return sum_c / sum_s


def get_fea_prob(clicks, selections):
    sum_c = pandas.DataFrame.sum(clicks, axis=0)
    sum_s = pandas.DataFrame.sum(selections, axis=0)
    return sum_c / sum_s


# def naive_decide(clicks, selections, user_feature, n_top=10):
#     user_feature = list(map(str, user_feature))
#     sum_c = pandas.DataFrame.sum(clicks[user_feature], axis=1)
#     sum_s = pandas.DataFrame.sum(selections[user_feature], axis=1)
#     score = sum_c / sum_s
#
#     return list(score.nlargest(n_top).index)


def naive_decide(prob_mat, fea_prob, adv_prob, user_feature, adv_id):
    user_feature = list(map(str, user_feature))
    s = pandas.DataFrame.sum(prob_mat.loc[:][user_feature] * fea_prob[user_feature]) / adv_prob[adv_id]
    print(s)

    exit()

    s = float(s)

    return s


def naive_prob(prob_mat, fea_prob, adv_prob, user_feature, adv_id):
    user_feature = list(map(str, user_feature))
    s = pandas.DataFrame.sum(prob_mat.loc[adv_id][user_feature] * fea_prob[user_feature]) / adv_prob[adv_id]
    s = float(s)

    return s


def naive_predict_thread(part_id, prob_mat, fea_prob, adv_prob, features_arr, adv_id_arr):
    n_data = len(adv_id_arr)
    with open("outputs/naive/t1_app_bayes_part_{i}.csv".format(i=part_id), "w") as fp:
        for i in range(n_data):
            fp.write("{:.5f}\n".format(naive_prob(prob_mat, fea_prob, adv_prob, features_arr[i], adv_id_arr[i])))


class naivePredictThread(threading.Thread):
    def __init__(self, part_id, prob_mat, fea_prob, adv_prob, features_arr, adv_id_arr):
        threading.Thread.__init__(self)
        self.part_id = part_id
        self.prob_mat = prob_mat
        self.fea_prob = fea_prob
        self.adv_prob = adv_prob
        self.features_arr = features_arr
        self.adv_id_arr = adv_id_arr

    def run(self):
        n_data = len(self.adv_id_arr)
        flag = n_data / 100
        with open("outputs/naive/parts/t1_app_bayes_part_{i}.csv".format(i=self.part_id), "w") as fp:
            for i in range(n_data):
                if i % flag == 0:
                    print("Thread-{tid}: {p}% Done".format(tid=self.part_id, p=i / flag))

                s = naive_prob(self.prob_mat, self.fea_prob, self.adv_prob, self.features_arr[i], self.adv_id_arr[i])
                if s >= 1:
                    s = 0.9
                fp.write("{:.5f}\n".format(s))


def naive_predict_parallel():
    with open("data/cache/clk_stat_for_fea_and_adv.json", 'r') as fp:
        clicks = DataFrame(json.loads(fp.read())).T.fillna(0)
    with open("data/cache/sel_stat_for_fea_and_adv.json", 'r') as fp:
        selections = DataFrame(json.loads(fp.read())).T.fillna(0)

        # print(clicks)
        # print(selections)

    prob_mat = clicks / selections
    fea_prob = get_fea_prob(clicks, selections)
    adv_prob = get_adv_prob(clicks, selections)

    train_paths, test_paths = w4_data_processing.get_origin_data_path()
    adv_id_arr, ts_arr, features_arr = w4_data_processing.read_data(test_paths, is_train_format=False, verbose=True)

    n_data = len(ts_arr)
    n_parts = 20
    n_step = int(n_data / n_parts)
    head = 0
    tail = head + n_step

    thread_pool = []

    for i in range(n_parts):
        thread_pool.append(
            naivePredictThread(i, prob_mat, fea_prob, adv_prob, features_arr[head:tail], adv_id_arr[head:tail]))

        head += n_step
        tail = head + n_step

    thread_pool.append(
        naivePredictThread(n_parts, prob_mat, fea_prob, adv_prob, features_arr[head:], adv_id_arr[head:]))

    for thread in thread_pool:
        thread.start()

    for thread in thread_pool:
        thread.join()


def naive_predict(prob=True, low_dim=False):
    with open("data/cache/clk_stat_for_fea_and_adv_ld.json", 'r') as fp:
        clicks = DataFrame(json.loads(fp.read())).T.fillna(0)
    with open("data/cache/sel_stat_for_fea_and_adv_ld.json", 'r') as fp:
        selections = DataFrame(json.loads(fp.read())).T.fillna(0)

    # print(clicks)
    # print(selections)

    prob_mat = clicks / selections
    fea_prob = get_fea_prob(clicks, selections)
    adv_prob = get_adv_prob(clicks, selections)

    # print(DataFrame.sum(prob_mat, axis=1))
    # print(DataFrame.sum(prob_mat, axis=0))
    #
    # exit()

    if low_dim:
        train_paths, test_paths = w4_data_processing.get_low_dim_origin_data_path()
    else:
        train_paths, test_paths = w4_data_processing.get_origin_data_path()
    adv_id_arr, ts_arr, features_arr = w4_data_processing.read_data(test_paths[0], is_train_format=False, verbose=True)

    n_data = len(ts_arr)
    one_ctr = 0

    with open("outputs/naive/t1_app_3.csv", "w") as fp:
        for i in range(n_data):
            if i % 48500 == 0 and i > 0:
                print("{p} done one_ctr={o}".format(p=i / 48500, o=one_ctr))

            if features_arr[i] == [9]:
                s = 0.03576
            else:
                if prob:
                    s = naive_prob(prob_mat, fea_prob, adv_prob, features_arr[i], adv_id_arr[i])
                else:
                    s = naive_decide(prob_mat, fea_prob, adv_prob, features_arr[i], adv_id_arr[i])
                if s > 1:
                    s = 0.9
                    one_ctr += 1
            fp.write("{:.5f}\n".format(s))


if __name__ == "__main__":
    # naive_count(True)
    # naive_count()
    # naive_predict_parallel()
    naive_predict(prob=False, low_dim=True)
    # naive_counter()
    # naive_count()
