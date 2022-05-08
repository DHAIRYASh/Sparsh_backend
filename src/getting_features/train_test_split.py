# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:37:45 2021

@author: Divy
"""
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.get_class import encode, decode
from src.utils.utils import fetch, save, data_and_models

path_f = os.path.join(data_and_models, "numpy_data")


def normalize(x):
    '''
    normalize the data
    '''
    xmax, xmin = x.max(), x.min()
    x = (x - xmin) / (xmax - xmin)
    return x


def class_balance(x, y):
    '''
    Removes class imbalance
    '''
    from random import randrange
    s = y.shape[1]
    num = np.zeros(s)
    for each in y:
        i = 0
        for e in each:
            if e == 1:
                num[i] = num[i] + 1
            i = i + 1
    min_ = min(num)
    i = 0
    for e in num:
        if e == min_:
            num = np.zeros(s).tolist()
            num[i] = 1
            num = np.asarray(num)
            print(f"{decode([num])[0]} is having minimum data of {min_}")
            break
        i = i + 1
    num = np.zeros(s)
    came_index = []
    x_b = []
    y_b = []
    classes = get_label_code()
    # boolean = True
    while min(num) != min_ or max(num) != min_:
        j = 0
        while j < len(num):
            if num[j] < min_:
                index = randrange(y.shape[0])
                # print(num)
                if index not in came_index:
                    if (y[index] == classes[j]).all():
                        x_b.append(x[index])
                        y_b.append(y[index])
                        came_index.append(index)
                        num[j] = num[j] + 1
            j = j + 1
    x_b = np.asarray(x_b)
    y_b = np.asarray(y_b)
    return x_b, y_b


def split_slice_data(slice_path):
    '''
    Split slice data into train and validation
    '''
    x, y = [fetch(e) for e in slice_path]
    x, y = class_balance(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x = None
    y = None
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    x_train_s = os.path.join(path_f, "x_train_s.npy")
    x_test_s = os.path.join(path_f, "x_test_s.npy")
    x_val_s = os.path.join(path_f, "x_val_s.npy")
    y_train_s = os.path.join(path_f, "y_train_s.npy")
    y_test_s = os.path.join(path_f, "y_test_s.npy")
    y_val_s = os.path.join(path_f, "y_val_s.npy")

    save(x_train_s, normalize(x_train))
    save(x_test_s, normalize(x_test))
    save(x_val_s, normalize(x_val))
    save(y_train_s, y_train)
    save(y_test_s, y_test)
    save(y_val_s, y_val)
    return None


def split_image_data(image_path):
    '''
    Split image data into train and validation
    '''
    x, y = [fetch(e) for e in image_path]
    x, y = class_balance(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x = None
    y = None
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

    x_train_s = os.path.join(path_f, "x_train_im.npy")
    x_test_s = os.path.join(path_f, "x_test_im.npy")
    x_val_s = os.path.join(path_f, "x_val_im.npy")
    y_train_s = os.path.join(path_f, "y_train_im.npy")
    y_test_s = os.path.join(path_f, "y_test_im.npy")
    y_val_s = os.path.join(path_f, "y_val_im.npy")

    save(x_train_s, normalize(x_train))
    save(x_test_s, normalize(x_test))
    save(x_val_s, normalize(x_val))
    save(y_train_s, y_train)
    save(y_test_s, y_test)
    save(y_val_s, y_val)
    return None


def driver_split(image_path, slice_path):
    '''
    Run the split functions
    '''
    split_image_data(image_path)
    split_slice_data(slice_path)
    return None


def get_label_code():
    '''
    Get one hot encoded labels
    '''
    dict_p = os.path.join(data_and_models, "dict", "dict_e.pkl")
    a_file = open(dict_p, "rb")
    output = pickle.load(a_file)
    output = list(output.keys())
    return encode(output, False)
