import w4_data_processing
import w4_plot
import numpy as np
import json
import os.path
import pandas as pd


def time_to_hour(t):
    hour = t[0]
    minute = t[1]
    hour += 1 if minute > 30 else 0
    return (hour) % 24


def get_click_rate_time_series(adv_id, preserve_guest=True):
    X, y, ts = w4_data_processing.read_train_tabular(adv_id, feature_filter=[0], preserve=preserve_guest, with_ts=True)

    time_freq_recorder = dict()

    for h in range(0, 24):
        time_freq_recorder[h] = dict()
        time_freq_recorder[h]["s"] = 0
        time_freq_recorder[h]["c"] = 0

    for i in range(len(y)):
        h = ts[i][0]
        time_freq_recorder[h]["s"] += 1
        time_freq_recorder[h]["c"] += int(y[i])

    for h in range(0, 24):
        if not time_freq_recorder[h]["s"] == 0:
            time_freq_recorder[h]["r"] = time_freq_recorder[h]["c"] / time_freq_recorder[h]["s"]
        else:
            time_freq_recorder[h]["r"] = 0

    return time_freq_recorder


def cache_click_rate_for_guest():
    with open("data/cache/adv_id_in_test_data.json", "r") as fp:
        adv_id_arr = json.loads(fp.read())

    click_rate_for_guest_recorder = dict()
    for adv_id in adv_id_arr:
        click_rate_for_guest_recorder[adv_id] = get_click_rate_time_series(adv_id)

    with open("data/cache/click_rate_for_guest_by_time.json", "w") as fp:
        json.dump(click_rate_for_guest_recorder, fp)


def plot_guest_data(adv_id_arr):
    if not isinstance(adv_id_arr, list):
        adv_id_arr = [adv_id_arr]

    canvas = w4_plot.W4Canvas()
    n_data = len(adv_id_arr)

    for i in range(n_data):
        for is_guest in [True, False]:
            adv_id = adv_id_arr[i]
            recorder = get_click_rate_time_series(adv_id, is_guest)
            r_ctr = [100 * recorder[i]["r"] for i in range(0, 24)]
            color = canvas.get_color(i)
            label = "{id}_{suf}".format(id=adv_id, suf="guest" if is_guest else "logger")
            style = "dashed" if is_guest else "solid"
            canvas.draw_line_chart_2d(range(0, 24), r_ctr, need_xticks=True, color=color, label=label, line_style=style)

    canvas.set_legend()

    canvas.set_title("CTR time series")
    canvas.set_x_label("time in one day (hour)")
    canvas.set_y_label("CTR (%)")

    canvas.froze()


def calculate_intersections():
    with open("data/cache/dif_adv_in_each_file.json", "r") as f:
        ctr_obj = json.loads(f.read())

    train_paths, test_paths = w4_data_processing.get_low_dim_origin_data_path()
    train_file_arr = []
    test_file_arr = []
    for train_file in train_paths:
        train_file_arr.append(train_file.split("/")[-1])
    for test_file in test_paths:
        test_file_arr.append(test_file.split("/")[-1])

    print(train_file_arr)
    print(test_file_arr)
    n_train = len(train_file_arr)
    n_test = len(test_file_arr)

    print(n_train, n_test)
    summary_matrix = np.zeros((n_train, n_test), dtype=float)
    for i in range(n_train):
        for j in range(n_test):
            train_key = train_file_arr[i]
            test_key = test_file_arr[j]
            summary_matrix[i, j] = len(
                set(ctr_obj[test_key]).intersection(set(ctr_obj[train_key]))) / len(ctr_obj[train_key])

    df = pd.DataFrame(summary_matrix, index=train_file_arr)
    df.columns = test_file_arr
    print(df)


def calculate_all_different_adv_in_each_dataset():
    train_paths, test_paths = w4_data_processing.get_low_dim_origin_data_path()

    ctr_obj = dict()
    for train_file in train_paths:
        print(train_file)
        adv_id_arr, click_truth_arr, ts_arr, features_arr = w4_data_processing.read_data(train_file,
                                                                                         is_train_format=True)
        file_name = train_file.split("/")[-1]
        ctr_obj[file_name] = list(set(adv_id_arr))
        print(ctr_obj[file_name])

    for test_file in test_paths:
        print(test_file)
        adv_id_arr, ts_arr, features_arr = w4_data_processing.read_data(test_file, is_train_format=False)
        file_name = test_file.split("/")[-1]
        ctr_obj[file_name] = list(set(adv_id_arr))

    with open("data/cache/dif_adv_in_each_file.json", "w") as f:
        json.dump(ctr_obj, f)


if __name__ == "__main__":
    plot_guest_data(["id-576772", "id-595770", "id-591728", "id-584027", "id-611482"])
    # calculate_all_different_adv_in_each_dataset()
    # calculate_intersections()
