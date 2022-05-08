import os
import pickle

import numpy as np
import skimage.io as io
from PIL import Image
from numpy import asarray
from tensorflow.keras.preprocessing import image

data_and_models = "C:\\ai_dermatology\\ai_dermatology\\data_and_models"
contour_image_folder = os.path.join(data_and_models, "contour_image")
mid_dir_path = os.path.join(data_and_models, "mid_data")  # mid_data_folder
dir_path = os.path.join(data_and_models, "data")  # Training data folder (output folder)
data_acc = os.path.join(data_and_models, "data_acc")  # Validation data folder
data_handle_dict = os.path.join(data_and_models, "dict", "data_handle_dict.pkl")


def save(path, data):
    '''
    Save numpy, image or model to disk
    '''
    if ".npy" in path:
        np.save(path, data)
    elif str(type(data)) == "<class 'numpy.ndarray'>":
        save_image(path, data)
    elif str(type(data)) == "<class 'dict'>":
        save_dict(path, data)
    else:
        save_model(path, data)
    return path


def fetch(path, size=None):
    '''
    Fetch numpy array, image or model from disk
    '''
    if size is None:
        if ".sav" in path and "model" in path:
            load_model(path)
            return load_model(path)
        elif ".npy" in path:
            return np.load(path)
        elif ".pkl" in path:
            return fetch_dict(path)
        else:
            load_image(path)
            return load_image(path)
    elif isinstance(size, tuple):
        img = image.load_img(path, target_size=size)
        img = image.img_to_array(img)
        return img
    else:
        print("Please enter tuple in size")


def load_image(path):
    '''
    Fetch image from disk
    '''
    image = Image.open(path)
    image = asarray(image)
    return image


def save_image(path, image):
    '''
    Save image to disk
    '''
    io.imsave(path, image, check_contrast=False)
    return path


def load_model(path):
    '''
    Fetch model from disk
    '''
    model = open(path, 'rb')
    loaded_model = pickle.load(model)
    model.close()
    return loaded_model


def save_model(path, model):
    '''
    Save model to disk
    '''
    pickle.dump(model, open(path, 'wb'))
    return path


def save_dict(dict_p, dic):
    '''
    Fetch dictionary from disk
    '''
    a_file = open(dict_p, "wb")
    pickle.dump(dic, a_file)
    a_file.close()
    return None


def fetch_dict(dict_p):
    '''
    Save dictionary to disk
    '''
    a_file = open(dict_p, "rb")
    output = pickle.load(a_file)
    a_file.close()
    return output
