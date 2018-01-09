from __future__ import print_function
import numpy as np
from hyperopt import Trials, STATUS_OK, tpe

import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

import Augmentor


def base(x_train, y_train, x_test, y_test):
    n_out = 2
    input_shape = 96
    batch_size = 32
    epochs = 10
    steps_per_epoch = len(x_train) // batch_size
    lossfun = 'categorical_crossentropy'
    optimizer = 'adam'
    metrics = ['accuracy']

    p = Augmentor.Pipeline()

    p.flip_left_right(probability=0.5)
    if conditional({choice([True, False])}):
        p.crop_random(probability=1, percentage_area=0.8)
        p.resize(probability=1, width=96, height=96)
    if conditional({choice([True, False])}):
        p.random_erasing(probability=0.5, rectangle_area=0.2)
    if conditional({choice([True, False])}):
        p.shear(probability=0.3, max_shear_left=2, max_shear_right=2)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    p.status()
    g = p.keras_generator_from_array(x_train, y_train, batch_size=batch_size)
    g = ((x / 255., y) for (x, y) in g)

    inputs = Input(shape=input_shape)
    x = inputs
    x = Conv2D(32, (3, 3))(x)
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
        loss=lossfun,
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

    #return dict(zip(['loss', 'status', 'model'], [-acc, STATUS_OK, model]))
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
