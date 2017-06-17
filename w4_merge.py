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
        adv_obj_arr = json.loads(fp.read())

    adv_obj_arr = reversed(sorted(adv_obj_arr.items(), key=lambda x: x[1]))
    for adv_obj in adv_obj_arr:
        store_obj = dict()
        print("Start predicting adv : {id} with weight {w}".format(id=adv_obj[0], w=adv_obj[1]))
        store_obj = predict_fun(adv_obj[0], store_obj, *args)
        with open("data/cache/tmp/{id}.json".format(id=adv_obj[0]), "w") as fp:
            json.dump(store_obj, fp)

    store_obj = dict()

    for adv_obj in adv_obj_arr:
        with open("data/cache/tmp/{id}.json".format(id=adv_obj[0]), "r") as fp:
            store_obj.update(json.loads(fp.read()))

    with open(output_file_path, "w") as fp:
        for i in range(4859648):
            fp.write("{p} \n".format(p=store_obj[str(i)]))


import multiprocessing


def merge_worker(adv_obj_arr, predict_fun):
    for adv_obj in adv_obj_arr:
        # print("Start predicting adv : {id} with weight {w}".format(id=adv_obj[0], w=adv_obj[1]))
        store_obj = dict()
        store_obj = predict_fun(adv_obj[0], store_obj)
        with open("data/cache/tmp/{id}.json".format(id=str(adv_obj[0])), "w") as fp:
            json.dump(store_obj, fp)


def merge_predict_parallel(output_file_path, predict_fun, *args):
    """

    :param output_file_path: 
    :param predict_fun: 
    :param args: [list]
        This parameter is for specified hyper-parameter function 
    :return: 
    """
    with open("data/cache/adv_id_in_test_data.json", "r") as fp:
        adv_obj_arr = json.loads(fp.read())

    adv_obj_arr = list(reversed(sorted(adv_obj_arr.items(), key=lambda x: x[1])))

    n_treads = 5
    worker_pool = []

    # 230
    n_data = 230

    n_batch = int(n_data / n_treads)
    head = 0
    tail = head + n_batch

    for _ in range(n_treads):
        worker_pool.append(multiprocessing.Process(target=merge_worker, args=(adv_obj_arr[head:tail], predict_fun)))
        head = tail
        tail = head + n_batch

    for worker in worker_pool:
        worker.start()

    for worker in worker_pool:
        worker.join()

    store_obj = dict()

    for adv_id in adv_obj_arr:
        with open("data/cache/tmp/{id}.json".format(id=adv_id[0]), "r") as fp:
            store_obj.update(json.loads(fp.read()))

    with open(output_file_path, "w") as fp:
        for i in range(4859648):
            fp.write("{p} \n".format(p=store_obj[str(i)]))
