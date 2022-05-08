import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler

from src.utils.get_class import get_num_classes


def model_maker(x_train):
    '''
    Build the global model
    '''
    n_class = get_num_classes()

    shape = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
    input = tf.keras.Input(shape=(shape,))
    x = layers.Dense(15, activation="relu")(input)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(n_class, activation="softmax")(x)

    model = tf.keras.Model(input, output)
    optimizer = tf.keras.optimizers.RMSprop(lr=1e-3)
    model.compile(optimizer, "categorical_crossentropy", metrics=["accuracy"])

    return model


def train_model(model, x, y):
    '''
    train the global model
    '''
    batch_size, epochs, verbose = 20, 20, 1
    x_train, x_valid = x
    y_train, y_valid = y
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_valid = x_valid.reshape(x_valid.shape[0], -1)
    initial_learning_rate = 1e-3

    def lr_exp_decay(epoch, lr):
        k = 1e-5
        return initial_learning_rate * math.exp(-k * epoch)

    model.fit(x_train, y_train, batch_size, epochs, callbacks=[LearningRateScheduler(lr_exp_decay)],
              verbose=verbose, validation_data=(x_valid, y_valid))
    return model


def predict(model, x):
    '''
    Get the prediction of the global model
    '''
    x_reshape = x.reshape(x.shape[0], -1)
    y_pred_ = model.predict(np.asarray(x_reshape))
    y_pred = np.argmax(y_pred_, axis=1)
    y_pred_prob = list()
    for i in range(len(y_pred)):
        y_pred_prob.append(y_pred_[i][y_pred[i]])
    return y_pred, y_pred_prob
