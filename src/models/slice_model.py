import numpy as np
from xgboost import XGBClassifier

from src.utils.get_class import encode, decode, lencode, ldecode


def model_maker():
    '''
    Build the slice model
    '''
    model = XGBClassifier()
    return model


def train_model(model, x, y):
    '''
    Train the slice model
    '''
    x_train, x_test = x
    y_train, y_test = y
    y_train = decode(y_train)
    y_test = decode(y_test)
    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = lencode(y_train)
    model = model.fit(x_train, y_train)
    return model


def predict(model, x):
    '''
    Get the prediction of the slice model
    '''
    r_x_test = x.reshape(x.shape[0], -1)
    y_pred = model.predict(np.asarray(r_x_test))
    y_pred = encode(ldecode(y_pred), False)
    y_pred = np.asarray(y_pred)
    return y_pred
