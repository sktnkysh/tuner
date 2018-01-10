import os
import shutil
import json
import Augmentor
from tuner import utils
from tuner.load_data import df_fromdir, load_fromdf, get_labels_fromdir


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


def exec_p(p, sampling_size):
    p.sample(sampling_size)


def augment_dir(src_dir, out_dir, sampling_size=10, condition_file='cond.json'):
    _augment_dir(src_dir, sampling_size, condition_file)
    augmentor_out_dir = os.path.join(src_dir, 'output')
    for fname in os.listdir(augmentor_out_dir):
        src_file = os.path.join(augmentor_out_dir, fname)
        dst_file = os.path.join(out_dir, fname)
        if os.path.isfile(src_file):
            shutil.move(src_file, dst_file)
    shutil.rmtree(augmentor_out_dir)


def augment_dataset_custom_p(p_root_dir, dst_dir, p, sampling_size=10):
    exec_p(p, sampling_size)
    augmentor_out_dir = os.path.join(p_root_dir, 'output')
    utils.mvtree(augmentor_out_dir, dst_dir)


def augment_dataset(src_dir, out_dir, sampling_size=10, condition_file='cond.json'):
    labels = get_labels_fromdir(src_dir)
    utils.mkdir(out_dir)
    print(src_dir, out_dir)
    for label in labels:
        read_dir = os.path.join(src_dir, label)
        write_dir = os.path.join(out_dir, label)
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


import subprocess
from tuner import net


def exec_hyperas(train_dir, validation_dir, model):
    template_fname = 'template_base_hyperas.py'
    with open(template_fname, 'r') as f:
        template_code = f.read()

    code_hyperas = template_code.format(
        train_dir=train_dir, validation_dir=validation_dir, model=model.__name__)

    fname = 'code_hyperas.py'
    with open(fname, 'w') as f:
        f.write(code_hyperas)

    subprocess.run(['python', fname])


if __name__ == '__main__':
    train_dir = '../examples/dataset/brain/train'
    validation_dir = '../examples/dataset/brain/validation'
    exec_hyperas(train_dir, validation_dir, net.aug)
