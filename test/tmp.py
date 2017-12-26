
# coding: utf-8

# In[1]:


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

from tuner import utils
import net


# In[2]:


def aug(x_train, y_train, x_test, y_test):
    n_out = 4
    input_shape = (96, 96, 3)
    batch_size = 32
    epochs = 10
    steps_per_epoch = len(x_train) // batch_size

    p = Augmentor.Pipeline()

    p.flip_left_right(probability=0.5)
    if conditional({{choice([True, False])}}):
        p.crop_random(probability=1, percentage_area=0.8)
        p.resize(probability=1, width=96, height=96)
    if conditional({{choice([True, False])}}):
        p.random_erasing(probability=0.5, rectangle_area=0.2)
    if conditional({{choice([True, False])}}):
        p.shear(probability=0.3, max_shear_left=2, max_shear_right=2)
    print('####################')
    p.status()
    print('####################')
    g = p.keras_generator_from_array(x_train, y_train, batch_size=batch_size)
    g = ((x / 255., y) for (x, y) in g)

    inputs = Input(shape=input_shape)
    x = inputs
    x = Conv2D({{choice([16, 32])}}, (3, 3))(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3))(x)
    x = Conv2D(64, (3, 3))(x)
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
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

    model.fit_generator(
        g,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_test, y_test),
        epochs=epochs,
        verbose=2,
    )
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# In[3]:


def data():
    train_dname = '../examples/dataset/brain/train'
    test_dname = '../examples/dataset/brain/validation'
    df = utils.df_fromdir(train_dname)
    df = utils.oversampling_df(df, 80)
    #x_train, y_train = utils.load_fromdf(df, resize=96)
    x_train, y_train = utils.load_fromdf(df, resize=96, rescale=1)
    df = utils.df_fromdir(test_dname)
    #x_test, y_test = utils.load_fromdf(df)
    x_test, y_test = utils.load_fromdf(df, resize=96, rescale=1)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


# In[4]:


model = net.aug


# In[5]:


best_run, best_model = optim.minimize(
    model=model, data=data, algo=tpe.suggest, max_evals=10, trials=Trials(),
    notebook_name='tune_augmentor')


# In[6]:


best_run


# In[7]:


def p_fromconds(conds):
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


# In[8]:


p_fromconds(best_run)


# In[14]:


def fit_generator(lossfun='categorical_crossentropy',
                  optimizer=keras.optimizers.rmsprop(lr=0.005, decay=1e-6),
                  metrics=['accuracy'],
                  batch_size=32,
                  epochs=1,
                  steps_per_epochs=None,
                  augmentor_conds=None):
    def wrap(tuning_aug):
        def forward(x_train, y_train, x_test, y_test):
            n_out = y_train.shape[-1]
            input_shape = x_train.shape[1:]
            p = p_fromconds(best_run)
            g = p.keras_generator_from_array(x_train, y_train, batch_size=args.bs)
            g = ( (x/255., y) for (x,y) in g)
            
            model = target_net(n_out=n_out, input_shape=input_shape)
            print('build model')
            model.compile(loss=lossfun, optimizer=optimizer, metrics=metrics)
            model.fit_generator(
                g,
                batch_size=batch_size,
                epochs=epochs,
                steps_per_epochs=steps_per_epochs if steps_per_epochs else len(x_train)//batch_size,
                verbose=2,
                validation_data=(x_test, y_test))
            score, acc = model.evaluate(x_test, y_test, verbose=0)
            print('Test accuracy:', acc)
            return {'loss': -acc, 'status': STATUS_OK, 'model': model}
        return forward
    return wrap


# In[18]:


model = fit_generator(augmentor_conds=best_run, epochs=100,
                     optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999))(net.tune)


# In[22]:


@fit_generator(
    augmentor_conds=best_run, epochs=100,
    optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999))
def tune(x_train, y_train, x_test, y_test):
    n_out = 4
    input_shape = (96, 96, 3)
    batch_size = 32
    epochs = 200
    steps_per_epoch = 100

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
    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
    )
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# In[23]:


best_run, best_model = optim.minimize(
    model=tune, data=data, algo=tpe.suggest, max_evals=10, trials=Trials(),
    notebook_name='tune_augmentor')

