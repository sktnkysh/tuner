
from __future__ import print_function
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe

import tensorflow as tf
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Conv2D
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Add
from keras.preprocessing.image import ImageDataGenerator

import Augmentor
import data
import utils

#X_train, Y_train, X_validation, Y_validation = data.orig_raw()


train_dname = '../data/orig/train'
test_dname = '../data/orig/validation'
x_train, y_train = utils.load_data_fromdir(train_dname, rescale=(96,96))
x_test, y_test = utils.load_data_fromdir(test_dname, rescale=(96,96))


p = Augmentor.Pipeline()
p.zoom(probability=0.2, min_factor=1.1, max_factor=1.3)
#p.shear(probability=0.2, max_shear_left=1, max_shear_right=1)
p.crop_by_size(probability=0.5, width=96, height=96)
g = p.keras_generator_from_array(x_train, y_train, batch_size=1)

# model
n_out = 4
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(96,96,3)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(n_out))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])
model.fit_generator(
    g,
    steps_per_epoch=100,
    epochs=200,
    verbose=1,
    validation_data=(x_test, y_test),
    validation_steps=20
    )
