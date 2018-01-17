import os
import json

from hyperopt import hp
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

import tensorflow as tf
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import BatchNormalization, Flatten, Dense
from keras.layers import Input, Conv2D, Convolution2D
from keras.layers import Activation, concatenate, Dropout
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.utils import np_utils, to_categorical
from keras.applications.imagenet_utils import _obtain_input_shape

import Augmentor

from tuner import net
from tuner import utils
from tuner import load_data


def data():
    train_dir = '{train_dir}' 
    test_dir = '{validation_dir}'
    resize = 96 
    rescale = 1 
    df = load_data.df_fromdir(train_dir)
    x_train, y_train = load_data.load_fromdf(df, resize=resize, rescale=rescale)
    df = load_data.df_fromdir(test_dir)
    x_test, y_test = load_data.load_fromdf(df, resize=resize, rescale=rescale)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


model = net.{model}

best_run, best_model = optim.minimize(
    model=model,
    data=data,
    algo=tpe.suggest,
    max_evals=10,
    trials=Trials(),
)

with open('cond.json', 'w') as f:
        json.dump(best_run, f)
