import os
import random
import shutil

import numpy as np

from src.getting_features.save_features import slice_path, y_path_s, image_path, y_path_m
from src.pre_process.data_augmentation import driver_aug
from src.utils.get_class import encode
from src.utils.utils import save, fetch, data_and_models, data_handle_dict


def data_scarcity_handling(mid_dir, final_dir):
    '''
    Checks for data scarcity and handles it via augmentation and removal
    '''
    if not os.path.exists(data_handle_dict):
        dic = {}
    else:
        dic = fetch(data_handle_dict)

    y = [os.path.join(mid_dir, e) for e in os.listdir(mid_dir)]
    b_y = [os.path.basename(e) for e in y]
    en_y = encode(b_y, False)
    list_keys = list(dic.keys())
    i = 0
    for e in y:
        if e in list_keys:
            if not dic[e]:
                if len(os.listdir(e)) >= 13:
                    dic[e] = True
                    remove_from_data(e)
        else:
            if len(os.listdir(e)) < 13:
                dic[e] = False
            else:
                dic[e] = True
        i = i + 1
    list_keys = list(dic.keys())

    save(data_handle_dict, dic)

    for e in list_keys:
        if not dic[e]:
            if "numpy_data" in os.listdir(data_and_models):
                remove_from_data(e)
            driver_aug_scarcity(e)
    try:
        shutil.rmtree(final_dir)
    except:
        pass
    shutil.copytree(mid_dir, final_dir)
    for e in y:
        if dic[e]:
            shutil.rmtree(e)
        else:
            files = [os.path.join(e, f) for f in os.listdir(e)]
            for f in files:
                if "aug_image_" in f:
                    os.remove(f)
    return None


def driver_aug_scarcity(path):
    '''
    Augments data if its cardinality is less than 13 and saves it
    '''
    image_paths = [os.path.join(path, e) for e in os.listdir(path)]
    i = 0
    while len(os.listdir(path)) < 13:
        aug_list = driver_aug(fetch(random.choice(image_paths)))
        aug_image = random.choice(aug_list)
        aug_image_path = os.path.join(path, f"aug_image_{i}.png")
        save(aug_image_path, aug_image)
        i = i + 1
    return None


def remove_from_data(path_e):
    '''
    Removes the augmented data once its cardinality is greater than 13
    '''
    encoded_e = encode([os.path.basename(path_e)], False)
    remove_from_numpy_and_save(encoded_e, image_path, y_path_m)
    remove_from_numpy_and_save(encoded_e, slice_path, y_path_s)
    return None


def remove_from_numpy_and_save(encoded_e, x_path, y_path):
    '''
    Removes augmented data from numpy array if its cardinality is greater than 13
    '''
    x = list(fetch(x_path))
    y = list(fetch(y_path))
    i = 0
    while i < len(y):
        if (y[i] == encoded_e).all():
            y.pop(i)
            x.pop(i)
            i = i - 1
        i = i + 1
    save(x_path, np.asarray(x))
    save(y_path, np.asarray(y))
    return None
