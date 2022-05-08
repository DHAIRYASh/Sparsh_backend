import os
import pickle

import numpy as np

from src.getting_features.save_features import get_image_features
from src.getting_features.save_features import get_slice_features
from src.models.global_model import predict as model1_predict
from src.models.slice_model import predict as model2_predict
from src.utils.get_class import get_class, lencode, ldecode


def normalize(x):
    '''
    Normalize the data
    '''
    x = [normalize_p(e) for e in x]
    return x


def normalize_p(x):
    '''
    Normalize each image in the dataset
    '''
    xmax, xmin = x.max(), x.min()
    x = (x - xmin) / (xmax - xmin)
    return x


def get_percentage(l):
    '''
    gets percentage of each class for slicing dataset to compare with global model
    '''
    p = list()
    le = l.shape[1]
    i = 0
    while i < le:
        p.append(0)
        i = i + 1
    for each in l:
        i = 0
        while i < le:
            if each[i] == 1:
                p[i] = p[i] + 1
            i = i + 1
    l1 = l.shape[0]
    p = [(e / l1) * 100 for e in p]
    p = np.asarray(p)
    return p


def get_pred_data(models, data_path, boolean):
    '''
    Loads the validation dataset
    '''
    image_model, model2 = models
    if boolean:
        y = list()
    x_s = list()
    x_i = list()
    x_i_prob = list()
    for each in data_path:
        if boolean:
            y.append(get_class(each))
        if boolean:
            slice_ = get_slice_features([each], False)
        else:
            slice_ = get_slice_features([each], False, False)
        slice_ = np.asarray(slice_)
        slice_ = model2_predict(model2, slice_)

        x_s.append(get_percentage(slice_))
        if boolean:
            image = get_image_features([each], False)
        else:
            image = get_image_features([each], False, False)
        image = np.asarray(image)
        image = model1_predict(image_model, image)
        x_i.append(image[0][0])
        x_i_prob.append(image[1][0])

    x_s = np.asarray(x_s)
    x_s = normalize_p(x_s)
    x_i = np.asarray(x_i)
    if boolean:
        y = np.asarray(lencode(y))
    else:
        y = None
    return x_s, x_i, x_i_prob, y


def predict(models, img_folder_path, boolean):
    '''
    Gives the prediction of the model on the validation dataset
    '''
    x_s, x_im, x_im_prob, y = get_pred_data(models, img_folder_path, boolean)
    x_s_pred = np.argmax(x_s, axis=1)
    x_s_pred_prob = list()
    for i in range(len(x_s_pred)):
        x_s_pred_prob.append(x_s[i][x_s_pred[i]])
    y_pred = list()
    prob = list()
    for i in range(len(x_s_pred_prob)):
        s = x_s_pred[i]
        im = x_im[i]
        if s == im:
            y_pred.append(s)
            prob.append((x_im_prob[i]+x_s_pred_prob[i])/2)
        else:
            if x_im_prob[i] > x_s_pred_prob[i]:
                y_pred.append(im)
                prob.append(x_im_prob[i])
            else:
                y_pred.append(s)
                prob.append(x_s_pred_prob[i])
    y_pred = np.asarray(y_pred)
    if boolean:
        return ldecode(y_pred), ldecode(y)
    else:
        return ldecode(y_pred), prob
