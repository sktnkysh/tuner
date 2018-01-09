
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


import data

#X_train, Y_train, X_validation, Y_validation = data.orig_raw()




# generator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.02,
        height_shift_range=0.02,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)
test_datagen = ImageDataGenerator(rescale=1./255)

#train_generator = train_datagen.flow(X_train, Y_train)
#validation_generator = test_datagen.flow(X_validation, Y_validation)

train_generator = train_datagen.flow_from_directory(
        '../data/orig/train',
        target_size=(96, 96),
        batch_size=32,
        class_mode='categorical'
        )
validation_generator = train_datagen.flow_from_directory(
        '../data/orig/validation',
        target_size=(96, 96),
        batch_size=32,
        class_mode='categorical'
        )


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
    train_generator,
    steps_per_epoch=100,
    epochs=200,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=20
    )
