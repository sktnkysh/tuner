import os

import numpy as np

from hyperopt import hp
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from tuner import net
import json


def search_model(data, result_file=None):

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

    best_run, best_model = optim.minimize(
        model=net.simplenet,
        data=data,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )


    fname = 'simplenet.hdf5'
    best_model.save(fname)
    return fname
