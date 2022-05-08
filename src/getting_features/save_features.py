# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:48:11 2021

@author: Divy
"""

import os

import numpy as np

from src.getting_features.get_features import get_vgg_features
from src.getting_features.train_test_split import normalize
from src.utils.get_class import encode
from src.utils.utils import fetch, save, data_and_models

path_f = os.path.join(data_and_models, "numpy_data")
slice_path = os.path.join(path_f, "slice.npy")
y_path_s = os.path.join(path_f, "y_s.npy")
image_path = os.path.join(path_f, "image.npy")
y_path_m = os.path.join(path_f, "y.npy")


def save_slice_data(path_, boolean, b):
    '''
    Save slice data in numpy array
    '''
    if not boolean:
        s_array, y = get_slice_array(path_, boolean, b)
        return normalize(s_array)
    lista = os.listdir(data_and_models)
    if os.path.basename(path_f) not in lista:
        os.mkdir(path_f)
    lista = os.listdir(os.path.join(path_f))
    slice_array_, y_s_ = get_slice_array(path_, boolean, b)
    if "slice.npy" not in lista:
        save(slice_path, slice_array_)
        save(y_path_s, y_s_)
    else:
        slice_array, y_s = fetch(slice_path), fetch(y_path_s)
        slice_array, y_s = np.append(slice_array, slice_array_, axis=0), np.append(y_s, y_s_, axis=0)
        os.remove(slice_path)
        os.remove(y_path_s)
        save(slice_path, slice_array)
        save(y_path_s, y_s)
    return slice_path, y_path_s


def get_slice_array(pathl, boolean, b):
    '''
    Gives array of vgg features and disease labels
    '''
    s_paths = []
    for each in pathl:
        each_ = os.path.join(each, "slice")
        files = next(os.walk(each_))[2]
        for e in files:
            s_paths.append(os.path.join(each_, e))
    s_array, y = get_vgg_features(s_paths, 32, boolean, None, b)
    if boolean:
        y = encode(y, True)
    return s_array, y


def get_slice_features(paths, boolean, b=True):
    '''
    Calls get_slice_array and save_slice_data to get features and returns its path
    '''
    if not boolean:
        slice_array = save_slice_data(paths, boolean, b)
        return slice_array
    slice_path, y_path = save_slice_data(paths, boolean, b)
    return slice_path, y_path


def save_image_data(path_, boolean, b):
    '''
    Save image data in numpy array
    '''
    if not boolean:
        image, y = get_et_array(path_, boolean, b)
        return normalize(image)
    lista = os.listdir(data_and_models)
    if os.path.basename(path_f) not in lista:
        os.mkdir(path_f)
    image_, y_ = get_et_array(path_, boolean, b)
    if "y.npy" not in os.listdir(path_f):
        save(image_path, image_)
        save(y_path_m, y_)
    else:
        image, y = fetch(image_path), fetch(y_path_m)
        image, y = np.append(image, image_, axis=0), np.append(y, y_, axis=0)
        os.remove(image_path)
        os.remove(y_path_m)
        save(image_path, image)
        save(y_path_m, y)
    return image_path, y_path_m


def get_et_array(pathl, boolean, b):
    '''
    Gives array of vgg features and disease labels
    '''
    image = []
    for each in pathl:
        file = next(os.walk(each))[2]
        for e in file:
            image.append(os.path.join(each, e))
    image, y = get_vgg_features(image, 512, boolean, "image", b)
    if boolean:
        y = encode(y, True)
    return image, np.asarray(y)


def get_image_features(path, boolean, b=True):
    '''
    Gives array of vgg features and disease labels
    '''
    if not boolean:
        image = save_image_data(path, boolean, b)
        return image
    image_path, y_path = save_image_data(path, boolean, b)
    return image_path, y_path
