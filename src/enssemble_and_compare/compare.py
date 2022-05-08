import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from src.enssemble_and_compare.enssemble import predict
from src.pre_process.preprocess import driver_preprocess_train as preprocess
from src.utils.utils import save, fetch, data_and_models

deployed_models_dir = os.path.join(data_and_models, "models_deployed")
trained_models_dir = os.path.join(data_and_models, "models")


def save_model(models, models_dir):
    '''
    Saves and deploy models
    '''
    if os.path.basename(models_dir) not in os.listdir(data_and_models):
        os.mkdir(models_dir)
    path = os.path.join(models_dir, "image_model")
    models[0].save(path, save_format='tf')
    path = os.path.join(models_dir, "slice_model2.sav")
    save(path, models[1])
    return None


def get_acc(models, data_dir):
    '''
    Returns F1 Score on validation data
    '''
    path_dir = os.path.join(data_and_models, "path")
    if os.path.basename(path_dir) not in os.listdir(data_and_models):
        os.mkdir(path_dir)
    p = os.path.join(path_dir, "path_acc.npy")
    if "path_acc.npy" not in os.listdir(path_dir):
        val_paths = preprocess(data_dir, False)
        val_paths = np.asarray(val_paths)
        save(p, val_paths)
    else:
        val_paths = fetch(os.path.join(path_dir, "path_acc.npy"))
        val_path_a = preprocess(data_dir, False)
        if val_path_a is not None:
            val_paths = np.concatenate((val_paths, val_path_a), axis=0)
            os.remove(p)
            save(p, val_paths)
    y_pred, y = predict(models, val_paths, True)
    acc = f1_score(y, y_pred, average="weighted")
    return acc


def load_models():
    '''
    Loads global and slice models from trained models directory
    '''
    if os.path.basename(deployed_models_dir) not in os.listdir(data_and_models):
        os.mkdir(deployed_models_dir)
        return None
    elif os.path.basename(deployed_models_dir) in os.listdir(data_and_models):
        models = list()
        for m in os.listdir(deployed_models_dir):
            if "image" in m:
                image_model = tf.keras.models.load_model(os.path.join(deployed_models_dir, m))
                models.append(image_model)
            elif "model2" in m:
                model2 = fetch(os.path.join(deployed_models_dir, m))
                models.append(model2)
        return models


def validate(current_models, new_models, data_dir):
    '''
    Updates global and slice models if new f1_score is higher
    '''
    acc_current = get_acc(current_models, data_dir)
    acc_new = get_acc(new_models, data_dir)
    print(f"Current f1_score is {acc_current}%")
    print(f"New f1_score is {acc_new}%")
    # f1 score instead of accuracy
    if acc_new > acc_current:
        save_model(new_models, deployed_models_dir)
        print("new models are deployed")
    save_model(new_models, trained_models_dir)
    print("Latest models are saved.")
    return None


def compare(new_models, val_data_path):
    '''
    Compares new models with global and slice models if present
    '''
    current_models = load_models()
    if current_models is None:
        acc_new = get_acc(new_models, val_data_path)
        save_model(new_models, deployed_models_dir)
        save_model(new_models, trained_models_dir)
        print(f"New f1_score is {acc_new * 100}%")
        print("models are deployed")
        return None
    validate(current_models, new_models, val_data_path)
    return None
