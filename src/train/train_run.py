# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:29:42 2021

@author: Divy
"""
import datetime
import json
import os
import shutil

import requests

from src.driver import driver_crop_train as crop, generate_val_data, preprocess, save_img_f, save_vgg_slices_asnp, \
    train_test_val_split, \
    model_maker_d, train, \
    comparison, handeling_data_scarcity
from src.utils.utils import mid_dir_path, dir_path, data_acc

date = datetime.datetime.strptime("2021-09-09", "%Y-%m-%d")


def run():
    '''
    Driver function for training
    '''
    global date
    from_to_date = date.strftime("%Y-%m-%d")
    print(from_to_date)

    # Api call and pass values to respective lists
    r = requests.post("https://localhost:5007/get_train_data_from_to", data={'from': from_to_date, 'to': from_to_date})
    data = json.loads(r.text)
    print(data)

    date = date + datetime.timedelta(days=1)

    image_path_list = list()  # image list from api
    json_path_list = list()  # json list from api
    if len(data['status']) == 0:
        return None
    for e in data["status"]:
        image_path_list.append(e[0])
        json_path_list.append(e[1])

    crop(image_path_list, json_path_list, mid_dir_path)
    generate_val_data(mid_dir_path, data_acc)
    handeling_data_scarcity(mid_dir_path, dir_path)
    paths = preprocess(dir_path, True)
    if len(paths) == 0:
        return None
    img_path = save_img_f(paths)
    s_path = save_vgg_slices_asnp(paths)
    train_test_val_split(img_path, s_path)
    models = model_maker_d()
    models = train(models)
    comparison(models, data_acc)

    # To remove contents of directory
    path, dirs, files = next(os.walk(dir_path))
    for each in dirs:
        each = os.path.join(dir_path, each)
        shutil.rmtree(each)
    return None

if __name__ == "__main__":
    run()
