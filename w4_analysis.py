import w4_data_processing
import w4_plot
import numpy as np
import json


def time_to_hour(t):
    hour = t[0]
    minute = t[1]
    hour += 1 if minute > 30 else 0
    return (hour) % 24


def get_click_rate_time_series(adv_id):
    X, y, ts = w4_data_processing.read_tabular_ts_by_adv_id(adv_id, feature_filter=[0], preserve=True, with_ts=True)

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
        adv_id = adv_id_arr[i]
        recorder = get_click_rate_time_series(adv_id)
        r_ctr = [recorder[i]["r"] for i in range(0, 24)]
        color = canvas.get_color(i)
        canvas.draw_line_chart_2d(range(0, 24), r_ctr, need_xticks=True, color=color, label=adv_id)

    canvas.set_legend()

    canvas.set_title("Click-rate time series")
    canvas.set_x_label("time in one day (hour)")
    canvas.set_y_label("click-rate (%)")

    canvas.froze()

