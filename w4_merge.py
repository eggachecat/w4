import json

counter = 0


def merge_predict(output_file_path, predict_fun, *args):
    """
    
    :param output_file_path: 
    :param predict_fun: 
    :param args: [list]
        This parameter is for specified hyper-parameter function 
    :return: 
    """
    with open("data/cache/adv_id_in_test_data.json", "r") as fp:
        adv_id_arr = json.loads(fp.read())
    store_obj = dict()
    for adv_id in adv_id_arr:
        print("Start predicting adv : {id}".format(id=adv_id))
        store_obj = predict_fun(adv_id, store_obj, *args)

    with open(output_file_path, "w") as fp:
        for i in range(4859648):
            fp.write("{p} \n".format(p=store_obj[i]))
