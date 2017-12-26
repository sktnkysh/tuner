
# coding: utf-8

# In[13]:


from __future__ import print_function
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

from tuner import utils
import net



# In[9]:
from easydict import EasyDict
args = EasyDict({
    'dataset_dir': '~/tuner/examples/dataset/brain', 
    'epoch' : 100,
    'lr' : 0.005,
    'gpu': 0,
    'out': 'result',
    'resume': '',
})
try:
    __file__.endswith('py')
    import argparse
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--dataset_dir', '-d',type=str, default=args.bs,
                        help='examples dataset/brain')
    parser.add_argument('--epoch', '-e', type=int, default=args.epoch,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--learningrate', '-l', dest='lr', type=float, default=args.lr,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=args.gpu,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default=args.out,
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default=args.resume,
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', dest='n_in', type=int, default=args.n_in,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()
except:
    print('no argsparse')
    pass





# In[11]:


with open('cond.json', 'w') as f:
    json.dump(best_run, f)

