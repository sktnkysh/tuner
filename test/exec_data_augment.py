
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


def data():
    train_dir = os.path.join(args.dataset_dir, 'train')
    val_dir = os.path.join(args.dataset_dir, 'val')
    df = utils.df_fromdir(train_dir)
    df = utils.oversampling_df(df, 80)
    #x_train, y_train = utils.load_fromdf(df, resize=96)
    x_train, y_train = utils.load_fromdf(df, resize=96, rescale=1)
    df = utils.df_fromdir(val_dir)
    #x_val, y_val = utils.load_fromdf(df)
    x_val, y_val = utils.load_fromdf(df, resize=96, rescale=1)

    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    return x_train, y_train, x_val, y_val


# In[10]:


best_run, best_model = optim.minimize(
    model=net.aug, data=data, algo=tpe.suggest, max_evals=10, trials=Trials(),
    notebook_name='tune_augmentor')


# In[11]:


with open('cond.json', 'w') as f:
    json.dump(best_run, f)


# In[12]:


def gen_p():
    with open('cond.json', 'r') as f:
        conds = json.load(f)
    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    if conds['conditional']:
        p.crop_random(probability=1, percentage_area=0.8)
        p.resize(probability=1, width=96, height=96)
    if conds['conditional_1']:
        p.random_erasing(probability=0.5, rectangle_area=0.2)
    if conds['conditional_2']:
        p.shear(probability=0.3, max_shear_left=2, max_shear_right=2) 
    return p


# In[14]:


def tune(x_train, y_train, x_test, y_test):
    
    def gen_p():
        with open('cond.json', 'r') as f:
            conds = json.load(f)
        p = Augmentor.Pipeline()
        p.flip_left_right(probability=0.5)
        if conds['conditional']:
            p.crop_random(probability=1, percentage_area=0.8)
            p.resize(probability=1, width=96, height=96)
        if conds['conditional_1']:
            p.random_erasing(probability=0.5, rectangle_area=0.2)
        if conds['conditional_2']:
            p.shear(probability=0.3, max_shear_left=2, max_shear_right=2) 
        return p
    n_out = y_train.shape[-1] 
    input_shape = (96, 96, 3)
    batch_size = 32
    epochs = 100
    steps_per_epoch = len(x_train) // batch_size
    lossfun='categorical_crossentropy'
    optimizer='Adam'
    metrics=['accuracy']
    g = gen_p()

    inputs = Input(shape=input_shape)
    x = inputs
    ch = {{choice([16, 32])}}
    x = Conv2D(ch, (3, 3))(x)
    x = Conv2D(ch, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    ch = {{choice([32, 64])}}
    x = Conv2D(ch, (3, 3))(x)
    x = Conv2D(ch, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_out)(x)
    x = Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss=lossfun, optimizer=optimizer, metrics=metrics)
    model.fit_generator(
        g,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# In[15]:


best_run, best_model = optim.minimize(
    model=tune, data=data, algo=tpe.suggest, max_evals=10, trials=Trials(),
    notebook_name='tune_augmentor')

