import json


def merge_predict(predict_fun, *args):
    with open("data/cache/adv_id_in_test_data.json", "r") as fp:
        adv_id_arr = json.loads(fp.read())
    store_obj = dict()
    for adv_id in adv_id_arr:
        print(adv_id)
        store_obj = predict_fun(*args)

    with open("outputs/prediction_ffm.csv", "w") as fp:
        for i in range(4859648):
            fp.write("{p} \n".format(p=store_obj[i]))
