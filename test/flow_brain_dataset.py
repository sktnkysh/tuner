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
from tuner import tune_cnn
from tuner import use_hyperas
from tuner import net
from tuner.dataset import ClassificationDataset, AugmentDataset

# In[2]:

brain_dataset = ClassificationDataset('./brain')

# In[3]:

min(brain_dataset.counts_train_data().values())

# In[4]:

aug_brain = AugmentDataset(brain_dataset)

# In[8]:

aug_brain.search_opt_augment(model=net.neoaug)

# In[12]:

aug_brain.augment_dataset()

# In[13]:

best_condition, best_model = use_hyperas.exec_hyperas(
    aug_brain.augmented_dir,
    aug_brain.dataset.validation_dir,
    net.simplenet,
    batch_size=32,
    epochs=100,
    optimizer='adam')
fname = 'simplenet.hdf5'
best_model.save(fname)
