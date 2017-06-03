import w4_data_processing
import os
import subprocess
import json
import w4_analysis


def vector_to_str_with_label(y, x_vec):
    row_str = str(y)
    for i in range(len(x_vec)):
        # in this case field is feature
        field = i
        row_str += " {field}:{field}:{val}".format(field=field, val=x_vec[i])

    return row_str


def vector_to_str(x_vec):
    row_str = "0"
    n_data = len(x_vec)
    for i in range(n_data):
        # in this case field is feature
        field = i
        row_str += " {field}:{field}:{val}".format(field=field, val=x_vec[i])


    return row_str


def train_ffm(adv_id):
    """
    run 
    cd /mnt/d/workstation/Python/w4nn4cry/data/ffm_format && ffm-train -p test.txt -l 0.00002 --auto-stop train.txt
    :return: 
    """
    X, y = w4_data_processing.read_tabular_ts_by_adv_id(adv_id, feature_filter=[0], preserve=False)

    folders = w4_data_processing.spilt_data(X, y, 10)

    for train_index, test_index in folders:
        # just once
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        root_path = "models/ffm/{id}".format(id=adv_id)
        os.mkdir(root_path)

        with open("{rp}/train.txt".format(rp=root_path), "w") as fp:
            X_train = X_train.toarray()
            for i in range(X_train.shape[0]):
                row = vector_to_str_with_label(y_train[i], X_train[i])
                fp.write("{row}\n".format(row=row))

        with open("{rp}/valid.txt".format(rp=root_path), "w") as fp:
            X_test = X_test.toarray()
            for i in range(X_test.shape[0]):
                row = vector_to_str_with_label(y_test[i], X_test[i])
                fp.write("{row}\n".format(row=row))

        bash_path = "C:\\Windows\\System32\\bash.exe"
        bash_command = "cd {rp} && ffm-train --quiet -p test.txt --auto-stop train.txt".format(rp=root_path)
        args = [bash_path, "-c", bash_command]
        subprocess.call(args)

        return


def predict_ffm(adv_id, guest_obj, store_obj):
    src_root_path = "data/tabular_ts_predict/{id}".format(id=adv_id)
    model_root_path = "models/ffm/{id}".format(id=adv_id)

    l_arr = []
    feature_arr = []

    with open("{rp}/test.txt".format(rp=src_root_path), 'r') as f:
        for line in f:
            components = str.split(line.rstrip(), " ", 2)
            l = int(components[0])
            feature = eval(components[2])
            if feature == [0]:
                h = components[1]
                store_obj[l] = guest_obj[adv_id][h]["r"]
            else:
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
def predict():
    print("predict")
    with open("data/cache/click_rate_for_guest_by_time.json", "r") as fp:
        guest_obj = json.loads(fp.read())

    with open("data/cache/adv_id_in_test_data.json", "r") as fp:
        adv_id_arr = json.loads(fp.read())
    store_obj = dict()
    for adv_id in adv_id_arr:
        print(adv_id)
        store_obj = predict_ffm(adv_id, guest_obj, store_obj)

    with open("outputs/prediction_ffm.csv", "w") as fp:
        for i in range(4859648):
            fp.write("{p} \n".format(p=store_obj[i]))


predict()

# with open("data/cache/adv_id_in_test_data.json", "r") as fp:
#     adv_id_arr = json.loads(fp.read())
#
# for adv_id in adv_id_arr:
#     print("train", adv_id)
#     train_ffm(adv_id)
