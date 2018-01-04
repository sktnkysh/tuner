import subprocess
from tuner import net
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

from tuner import utils


def exec_hyperas(train_dir, validation_dir, model):
    template_fname = 'hyperas_template_data.py'
    with open(template_fname, 'r') as f:
        template_code = f.read()

    fname = 'hyperas_data.py'
    with open(fname, 'w') as f:
        code = template_code.format(train_dir=train_dir, validation_dir=validation_dir)
        f.write(code)

    from hyperas_data import data

    best_condition, best_model = optim.minimize(
        model=model,
        data=data,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )
    return best_condition, best_model
