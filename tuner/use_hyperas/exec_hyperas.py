import subprocess
import json

import numpy as np

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
from tuner import load_data
from tuner import net


def exec_hyperas(\
        train_dir, validation_dir, model,
        resize=96, rescale=1, batch_size=32, epochs=10,
        loss='categorical_crossentropy',
        optimizer='adam'):
    template_fname = 'hyperas_template_data.py'
    with open(template_fname, 'r') as f:
        template_code = f.read()

    fname = 'hyperas_data.py'
    with open(fname, 'w') as f:
        code = template_code.format(train_dir=train_dir, validation_dir=validation_dir)
        f.write(code)

    from hyperas_data import data
    df = utils.df_fromdir(validation_dir)
    x_test, y_test = load_data.load_fromdf(df, resize=resize, rescale=rescale)

    params = {
        'n_out': np.unique(y_test).size,
        'input_shape': x_test.shape[1:],
        'batch_size': batch_size,
        'epochs': epochs,
        'lossfun': loss,
        'optimizer': optimizer
    }
    with open('tmp_params.json', 'w') as f:
        json.dump(params, f)

    best_condition, best_model = optim.minimize(
        model=model,
        data=data,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )
    return best_condition, best_model
