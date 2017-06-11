import w4_data_processing
import os
import subprocess
import json
import w4_analysis
import w4_merge


# def vector_to_str_with_label(y, x_vec):
#     row_str = str(y)
#     for i in range(len(x_vec)):
#         # in this case field is feature
#         field = i
#         row_str += " {field}:{field}:{val}".format(field=field, val=x_vec[i])
#
#     return row_str


def vector_to_str(x_vec, y=None):
    if y is None:
        row_str = "0"
    else:
        row_str = str(y)
    n_data = len(x_vec)
    for i in range(n_data):
        # in this case field is feature
        field = i
        row_str += " {field}:{field}:{val}".format(field=field, val=x_vec[i])

    return row_str


def train_by_adv_id(adv_id, with_guest_filtered=False):
    """
    run 
    cd /mnt/d/workstation/Python/w4nn4cry/data/ffm_format && ffm-train -p test.txt -l 0.00002 --auto-stop train.txt
    :return: 
    """
    if with_guest_filtered:
        X, y = w4_data_processing.read_train_tabular(adv_id, feature_filter=[0], preserve=False)
    else:
        X, y = w4_data_processing.read_train_tabular(adv_id)

    folders = w4_data_processing.spilt_data(X, y, 3)

    for train_index, test_index in folders:
        # just once
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        root_path = "models/ffm/{id}".format(id=adv_id)
        os.mkdir(root_path)

        with open("{rp}/train.txt".format(rp=root_path), "w") as fp:
            X_train = X_train.toarray()
            for i in range(X_train.shape[0]):
                row = vector_to_str(X_train[i], y_train[i])
                fp.write("{row}\n".format(row=row))

        with open("{rp}/valid.txt".format(rp=root_path), "w") as fp:
            X_test = X_test.toarray()
            for i in range(X_test.shape[0]):
                row = vector_to_str(X_test[i], y_test[i])
                fp.write("{row}\n".format(row=row))

        bash_path = "C:\\Windows\\System32\\bash.exe"
        bash_command = "cd {rp} && ffm-train --quiet -p valid.txt --auto-stop train.txt".format(rp=root_path)
        args = [bash_path, "-c", bash_command]
        subprocess.call(args)

        return


def predict_by_adv_id(adv_id, store_obj):
    src_root_path = "data/tabular_ts_predict/{id}".format(id=adv_id)
    model_root_path = "models/ffm/{id}".format(id=adv_id)

    l_arr = []
    feature_arr = []

    with open("{rp}/test.txt".format(rp=src_root_path), 'r') as f:
        for line in f:
            components = str.split(line.rstrip(), " ", 2)
            l = int(components[0])
            feature = eval(components[2])
            l_arr.append(l)
            feature_arr.append(feature)

    with open("{rp}/predict.txt".format(rp=model_root_path), "w") as fp:
        for feature in feature_arr:
            fp.write("{row}\n".format(row=vector_to_str(feature)))

    bash_path = "C:\\Windows\\System32\\bash.exe"
    # bash_command_path = "/mnt/d/workstation/Python/w4nn4cry/models/ffm/{id}".format(id=adv_id)
    bash_command = "cd {rp} && ffm-predict predict.txt train.txt.model output.txt".format(rp=model_root_path)
    args = [bash_path, "-c", bash_command]
    subprocess.call(args)

    output_arr = []
    with open("{rp}/output.txt".format(rp=model_root_path), "r") as f:
        for line in f:
            output_arr.append(float(line.strip()))

    for i in range(len(l_arr)):
        store_obj[l_arr[i]] = output_arr[i]

    return store_obj


def train():
    with open("data/cache/adv_id_in_test_data.json", "r") as fp:
        adv_id_arr = json.loads(fp.read())
    for adv_id in adv_id_arr:
        print("Start training {id}".format(id=adv_id))
        train_by_adv_id(adv_id)


def predict():
    w4_merge.merge_predict("outputs/prediction_ffm.csv", predict_by_adv_id)


#
# def predict():
#     print("predict")
#     with open("data/cache/click_rate_for_guest_by_time.json", "r") as fp:
#         guest_obj = json.loads(fp.read())
#
#     with open("data/cache/adv_id_in_test_data.json", "r") as fp:
#         adv_id_arr = json.loads(fp.read())
#     store_obj = dict()
#     for adv_id in adv_id_arr:
#         print(adv_id)
#         store_obj = predict_ffm(adv_id, guest_obj, store_obj)
#
#     with open("outputs/prediction_ffm.csv", "w") as fp:
#         for i in range(4859648):
#             fp.write("{p} \n".format(p=store_obj[i]))


if __name__ == "__main__":
    predict()

# predict()

# with open("data/cache/adv_id_in_test_data.json", "r") as fp:
#     adv_id_arr = json.loads(fp.read())
#
# for adv_id in adv_id_arr:
#     print("train", adv_id)
#     train_ffm(adv_id)
