import numpy as np
import glob
import json
import w4_data_processing
import w4_evalution
import time

for file_path in glob.glob("data/cache/p_tmp_2/*.json"):
    adv_id = file_path.split("\\")[-1].split(".")[0]
    with open(file_path, "r") as fp:
        obj = json.loads(fp.read())
    obj = list(sorted(obj.items(), key=lambda x: int(x[0])))
    y_pred = [item[1] for item in obj]
    X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=None, w=True)
    logloss = w4_evalution.get_logloss(y, y_pred)

    print(adv_id, logloss)

# obj = dict()
# with open("data/cache/p_tmp/id-605378.json", "r") as fp:
#     obj = json.loads(fp.read())
#
# obj = list(sorted(obj.items(), key=lambda x: int(x[0])))
# y_pred = [item[1] for item in obj]
#
# benchmark = w4_evalution.get_out_benchmark(adv_id)
# X, y = w4_data_processing.read_train_tabular(adv_id, low_dim=False, group_map=None, w=True)
# logloss = w4_evalution.get_logloss(y, y_pred)
#
# print(adv_id, logloss, benchmark)



# e = time.time()
#
# print(e - s)

# s = time.time()
# for file_path in glob.glob("data/cache/p_tmp_2/*.json"):
#     adv_id = file_path.split("\\")[-1].split(".")[0]
#     store_obj = dict()
#     with open(file_path, "r") as fp:
#         store_obj.update(json.loads(fp.read()))
# e = time.time()
# print(e - s)
