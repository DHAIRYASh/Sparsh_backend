# -*- coding: utf-8 -*-
"""
Created on Mon May  3 17:29:08 2021

@author: vyass
"""

import os

import numpy as np

from src.enssemble_and_compare.compare import compare
from src.getting_features.save_features import get_image_features
from src.getting_features.save_features import get_slice_features
from src.getting_features.train_test_split import driver_split
from src.models.global_model import model_maker as model_maker1, train_model as train_model1
from src.models.slice_model import model_maker as model_maker2, train_model as train_model2
from src.pre_process.crop import driver_crop_pred, driver_crop_train
from src.pre_process.handel_data_scarcity import data_scarcity_handling
from src.pre_process.preprocess import driver_preprocess_train
from src.train.make_val_data import gen_val_data
from src.utils.utils import fetch, save, data_and_models

# from connector import login, get_data, extract_data

path_f = os.path.join(data_and_models, "numpy_data")
x_train_im = os.path.join(path_f, "x_train_im.npy")
x_test_im = os.path.join(path_f, "x_test_im.npy")
x_val_im = os.path.join(path_f, "x_val_im.npy")
y_train_im = os.path.join(path_f, "y_train_im.npy")
y_test_im = os.path.join(path_f, "y_test_im.npy")
y_val_im = os.path.join(path_f, "y_val_im.npy")
x_train_s = os.path.join(path_f, "x_train_s.npy")
x_test_s = os.path.join(path_f, "x_test_s.npy")
x_val_s = os.path.join(path_f, "x_val_s.npy")
y_train_s = os.path.join(path_f, "y_train_s.npy")
y_test_s = os.path.join(path_f, "y_test_s.npy")
y_val_s = os.path.join(path_f, "y_val_s.npy")


# Calls connector and its functions to extract the data
# input - strat_date,end_date
# output - parent folder path

# def fetch_data(start_date,end_date):
#     login()
#     data_location=get_data(start_date,end_date)
#     parent_folder_path=extract_data(data_location)
#     return parent_folder_path


def crop_pred(annotation_folder, dir_path):
    '''
    Calls the crop function to crop the images and save them in the specified directory(for prediction pipeline)
    '''
    exp_list = driver_crop_pred(annotation_folder, dir_path)
    return exp_list


def crop_train(image_path_list, json_path_list, folder_out):
    '''
    Calls the crop function to crop the images and save them in the specified directory(for training pipeline)
    '''
    driver_crop_train(image_path_list, json_path_list, folder_out)
    return None


def generate_val_data(mid_dir, val_dir):
    '''
    Calls the gen_val_data function to generate the validation data
    '''
    gen_val_data(mid_dir, val_dir)
    return None


def handeling_data_scarcity(mid_dir, final_dir):
    '''
    Calls the data_scarcity_handling function to handle the data scarcity
    '''
    data_scarcity_handling(mid_dir, final_dir)
    return None


def preprocess(parent_folder_path, boolean):
    '''
    Preprocess the data and converts it to 512by512 images and 32by32 slices
    '''
    paths = driver_preprocess_train(parent_folder_path, boolean)
    if "path" not in os.listdir(data_and_models):
        os.mkdir(os.path.join(data_and_models, "path"))
    p = os.path.join(data_and_models, "path", "folder_path.npy")
    save(p, np.asarray(paths))
    return paths


def save_vgg_slices_asnp(paths):
    '''
    Saves vgg features of slices as numpy array
    '''
    slice_path, y_path = get_slice_features(paths, True)
    if "path" not in os.listdir(data_and_models):
        os.mkdir(os.path.join(data_and_models, "path"))
    p = os.path.join(data_and_models, "path", "s_path.npy")
    save(p, np.asarray([slice_path, y_path]))
    return slice_path, y_path


def save_img_f(paths):
    '''
    Save image features as numpy array
    '''
    image_path, y_path = get_image_features(paths, True)
    if "path" not in os.listdir(data_and_models):
        os.mkdir(os.path.join(data_and_models, "path"))
    p = os.path.join(data_and_models, "path", "im_path.npy")
    save(p, np.asarray([image_path, y_path]))
    return image_path, y_path


def train_test_val_split(et_path, s_path):
    '''
    Train and validation split
    '''
    return driver_split(et_path, s_path)


def model_maker_d():
    '''
    Builds global and slice model
    '''
    image_model = model_maker1(fetch(x_train_im))
    model2 = model_maker2()
    models = [image_model, model2]
    print("models are ready to train")
    return models


# Calls Training funtions of models along with ensemmble function to train all model
# input - slice_data, threshold, edge
# output - y_test_pred2, models

def train(models):
    '''
    Trains global and slice model
    '''
    # image_model, model2, slice_vote = models
    image_model, model2 = models

    image_model = train_model1(image_model, [fetch(x_train_im), fetch(x_val_im)], [fetch(y_train_im), fetch(y_val_im)])
    model2 = train_model2(model2, [fetch(x_train_s), fetch(x_test_s)], [fetch(y_train_s), fetch(y_test_s)])
    models = [image_model, model2]
    print("models are trained")
    return models


# #Predicts the output on validation data by calling predict function
# #input - x_val, y_val, models
# #output - accuracy

def comparison(new_models, val_data_path):
    '''
    Compares the output of new models with old models
    '''
    compare(new_models, val_data_path)
    return None
