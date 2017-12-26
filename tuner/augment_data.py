import os
import json
import Augmentor
from tuner import utils
from tuner.load_data import df_fromdir, load_fromdf


def _augment_dir(src_dir, sampling_size=10, condition_file='cond.json'):
    with open(condition_file, 'r') as f:
        conds = json.load(f)
    p = Augmentor.Pipeline(src_dir)
    p.flip_left_right(probability=0.5)
    if conds['conditional']:
        p.crop_random(probability=1, percentage_area=0.8)
        p.resize(probability=1, width=96, height=96)
    if conds['conditional_1']:
        p.random_erasing(probability=0.5, rectangle_area=0.2)
    if conds['conditional_2']:
        p.shear(probability=0.3, max_shear_left=2, max_shear_right=2)
    p.sample(sampling_size)


def augment_dir(src_dir, out_dir, sampling_size=10, condition_file='cond.json'):
    #if os.path.exists(out_dir):
    #    raise 'exsists {}'.format(out_dir)
    _augment_dir(src_dir, sampling_size, condition_file)
    utils.mvtree(os.path.join(src_dir, 'output'), out_dir)


def augment_dataset(src_dir, out_dir, sampling_size=10, condition_file='cond.json'):
    labels = os.listdir(src_dir)
    utils.mkdir(out_dir)
    for label in labels:
        read_dir = os.path.join(src_dir, label)
        write_dir = os.path.join(out_dir, label)
        #utils.mkdir(write_dir)
        augment_dir(read_dir, write_dir, sampling_size, condition_file)


import numpy as np

from hyperopt import hp
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from tuner import net
import json


def search_condition(data, result_file='cond.json'):

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
        model=net.aug,
        data=data,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )

    with open(result_file, 'w') as f:
        json.dump(best_run, f)
        print('{} dump.'.format(result_file))
