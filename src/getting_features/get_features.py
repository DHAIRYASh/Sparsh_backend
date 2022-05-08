# -*- coding: utf-8 -*-
"""
Created on Wed May  5 21:12:09 2021

@author: Divy
"""
import os
from pathlib import Path

import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

from src.utils.utils import save, fetch

model = VGG19(weights="imagenet", include_top=False)


def get_vgg_features(p_list, s, boolean, i, b):
    '''
    Gives VGG features for a given image paths and its category
    '''
    folder_path = None
    feature = []
    y = list()
    path_list = []
    for e in p_list:
        path_list.append(e)
    if i is not None:
        for e in p_list:
            if ".npy" in e:
                feature.append(fetch(e))
                if boolean:
                    y.append(Path(e).parent.absolute())
                path_list.remove(e)
                k = e.split(".")[0] + ".png"
                path_list.remove(k)
    else:
        for e in p_list:
            if ".npy" in e:
                path_list.remove(e)
                k = os.path.join(os.path.dirname(e), "s" + e.split(".")[0].split("slice")[1][1:] + ".png")
                path_list.remove(k)
                feature.append(fetch(e))
                if boolean:
                    y.append(Path(Path(e).parent.absolute()).parent.absolute())

    for e in path_list:
        x = fetch(e, (s, s))
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        x = model.predict(x)
        x = x[0]
        if boolean or b:
            folder_path_temp = str(Path(e).parent.absolute())
            if folder_path != folder_path_temp:
                folder_path = folder_path_temp
            if i is None:
                num = e.split(".")[0].split("slice")[1].split("s")[1]
                path = os.path.join(folder_path, f"{num}.npy")
            else:
                path = os.path.join(folder_path, f"{i}.npy")
            # print("saving features")
            save(path, x)
        feature.append(x)
        if i is None:
            if boolean:
                y.append(Path(Path(e).parent.absolute()).parent.absolute())
        else:
            if boolean:
                y.append(Path(e).parent.absolute())
    feature = np.asarray(feature)
    if boolean:
        y = np.asarray(y)
    return feature, y
