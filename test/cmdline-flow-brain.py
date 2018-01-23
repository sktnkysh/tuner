# coding: utf-8

# In[1]:

import os
import subprocess

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
import json

import tuner
from tuner import utils
from tuner import load_data
from tuner import augment_data
from tuner import use_hyperas
from tuner import net
from tuner.dataset import ClassificationDataset, AugmentDataset

from datetime import datetime

# In[2]:

from easydict import EasyDict
params = EasyDict({
    'src_dir': '../micin-dataset/brain',
    'dst_dir': './brain',
    'bs': 32,
    'epochs': 10,
    'save_model_file': 'tuner.{}.model.hdf5'.format(int(datetime.now().timestamp()))
})
try:
    __file__
    import argparse
    argparse = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', dest='src_dir', type=str, help='dataset directory')
    parser.add_argument(
        '-o',
        '--output',
        dest='save_model_file',
        type=str,
        default=params.save_model_file,
        help='dataset directory')
    parser.add_argument(
        '-b', '--batchsize', dest='bs', type=int, default=params.bs, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=params.epochs, help='epochs')
    args = parser.parse_args()
except:
    args = params

# ### Convert  dataset from brain-dir to ready-dir

# In[3]:

df = load_data.df_fromdir_brain(args.src_dir)

# In[4]:

load_data.classed_dir_fromdf(df, "tmp/brain")

# ### Load dataset via ready-dir

# In[5]:

brain = ClassificationDataset('tmp/brain')

# In[6]:

brain = AugmentDataset(brain)

# In[7]:

brain.augmented_dir

# In[8]:

brain.dataset.train_dir

# ### Search best condition of data augmentation

# In[9]:

brain.search_opt_augment(model=net.neoaug)

# ### Execute data augmentation with best condition

# In[11]:

brain.augment_dataset()

# ### Tuning hyper parameter of CNN

# In[ ]:

best_condition, best_model = use_hyperas.exec_hyperas(
    brain.augmented_dir,
    brain.dataset.validation_dir,
    net.simplenet,
    batch_size=32,
    epochs=10,
    optimizer='adam',
    rescale=1. / 255)
best_model.save(args.save_model_file)
