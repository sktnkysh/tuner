#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import subprocess

import numpy as np
#
#from hyperopt import hp
#from hyperopt import Trials, STATUS_OK, tpe
#from hyperas import optim
#from hyperas.distributions import choice, uniform, conditional
#
#import tensorflow as tf
#import keras
#from keras import backend as K
#from keras.datasets import mnist
#from keras.layers import BatchNormalization, Flatten, Dense
#from keras.layers import Input, Conv2D, Convolution2D
#from keras.layers import Activation, concatenate, Dropout
#from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
#from keras.models import Sequential, Model
#from keras.utils import np_utils, to_categorical
#from keras.applications.imagenet_utils import _obtain_input_shape
#
#import Augmentor
#import json

import tuner
from tuner import utils
from tuner import load_data
from tuner import augment_data
from tuner import use_hyperas
from tuner import net
from tuner.dataset import ClassificationDataset, AugmentDataset

from datetime import datetime

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('src_dir', nargs='?', help='dataset directory')
parser.add_argument(
    '-o',
    '--output',
    dest='save_model_file',
    type=str,
    default='models/tuner.{}.model.hdf5'.format(int(datetime.now().timestamp())),
    help='dataset directory')
parser.add_argument('-b', '--batchsize', dest='bs', type=int, default=32, help='batch size')
parser.add_argument('-e', '--epochs', type=int, default=10, help='epochs')
args = parser.parse_args()

if args.src_dir:
    src_dir = args.src_dir
elif not sys.stdin.isatty():
    src_dir = sys.stdin.read().rstrip()
else:
    parser.print_help()

print(src_dir)
brain = ClassificationDataset(src_dir)

brain = AugmentDataset(brain)

### Search best condition of data augmentation
brain.search_opt_augment(model=net.neoaug)

### Execute data augmentation with best condition
brain.augment_dataset()

### Tuning hyper parameter of CNN
best_condition, best_model = use_hyperas.exec_hyperas(
    brain.augmented_dir,
    brain.validation_dir,
    net.simplenet,
    batch_size=args.bs,
    epochs=args.epochs,
    optimizer='adam',
    rescale=1. / 255)
best_model.save(args.save_model_file)
print('saved best model', args.save_model_file)
shutil.rmtree('standard_datasets')
