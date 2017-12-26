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
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    p.status()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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


def fire_module(x, squeeze=16, expand=64, activation='relu'):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    with tf.variable_scope('fire_module'):
        x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
        x = Activation(activation)(x)

        left = Convolution2D(expand, (1, 1), padding='valid')(x)
        left = Activation(activation)(left)

        right = Convolution2D(expand, (3, 3), padding='same')(x)
        right = Activation(activation)(right)

        x = concatenate([left, right], axis=channel_axis)
    return x


def drop(x_train, y_train, x_test, y_test):
    include_top = True
    input_tensor = None
    input_shape = (128, 128, 3)
    pooling = None
    classes = y_test.shape[-1]

    relu = 'relu'

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=227,
        min_size=48,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(
        16,
        (3, 3),
        strides=(2, 2),
        padding='valid',
        activation=relu,
    )(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Dropout({{uniform(0.1, 0.9)}})(x)

    squeeze = 16  # {{choice([16, 32])}}
    expand = 48  #{{choice([48, 64])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Dropout({{uniform(0.1, 0.9)}})(x)

    squeeze = 16  #{{choice([16, 32, 64])}}
    expand = 64  #{{choice([64, 128, 256])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Dropout({{uniform(0.1, 0.9)}})(x)

    squeeze = 24  #{{choice([24, 48, 96])}}
    expand = 96  #{{choice([96, 192, 384])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    squeeze = 32  #{{choice([32, 64, 128])}}
    expand = 128  #{{choice([128, 256, 512])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = Dropout({{uniform(0.1, 0.9)}})(x)

    x = Convolution2D(classes, (1, 1), padding='valid', activation=relu)(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of .
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=64,  # {{choice([32, 64, 128])}},
        epochs=100,
        verbose=2,
        validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def squeezenet(x_train, y_train, x_test, y_test):
    include_top = True
    input_tensor = None
    input_shape = (128, 128, 3)
    pooling = None
    classes = y_test.shape[-1]

    relu = {{choice(['relu', 'elu', 'selu'])}}

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=227,
        min_size=48,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5), (7, 7)])}},
        strides=(2, 2),
        padding='valid',
        activation=relu,
    )(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    squeeze = {{choice([16, 32])}}
    expand = {{choice([48, 64])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    squeeze = {{choice([16, 32, 64])}}
    expand = {{choice([64, 128, 256])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    squeeze = {{choice([24, 48, 96])}}
    expand = {{choice([96, 192, 384])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    squeeze = {{choice([32, 64, 128])}}
    expand = {{choice([128, 256, 512])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)

    x = Dropout({{uniform(0, 1)}})(x)

    x = Convolution2D(
        classes,
        (1, 1),
        padding='valid',
        activation=relu,
    )(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of .
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=64,  # {{choice([32, 64, 128])}},
        epochs=100,
        verbose=2,
        validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def squeezedrop(x_train, y_train, x_test, y_test):
    include_top = True
    input_tensor = None
    input_shape = (128, 128, 3)
    pooling = None
    classes = y_test.shape[-1]

    relu = {{choice(['relu', 'elu', 'selu'])}}

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=227,
        min_size=48,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5), (7, 7)])}},
        strides=(2, 2),
        padding='valid',
        activation=relu,
    )(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Dropout({{uniform(0, 1)}})(x)

    squeeze = {{choice([16, 32])}}
    expand = {{choice([48, 64])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Dropout({{uniform(0, 1)}})(x)

    squeeze = {{choice([16, 32, 64])}}
    expand = {{choice([64, 128, 256])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Dropout({{uniform(0, 1)}})(x)

    squeeze = {{choice([24, 48, 96])}}
    expand = {{choice([96, 192, 384])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    squeeze = {{choice([32, 64, 128])}}
    expand = {{choice([128, 256, 512])}}
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = fire_module(x, squeeze=squeeze, expand=expand)
    x = Dropout({{uniform(0, 1)}})(x)

    x = Convolution2D(classes, (1, 1), padding='valid', activation=relu)(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of .
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size=64,  # {{choice([32, 64, 128])}},
        epochs=100,
        verbose=2,
        validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def mysqueeze(x_train, y_train, x_test, y_test):
    from keras.applications.imagenet_utils import _obtain_input_shape
    from keras import backend as K
    from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
    from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
    from keras.models import Model
    from keras.engine.topology import get_source_inputs
    from keras.utils import get_file
    from keras.utils import layer_utils

    def fire_module(x, squeeze=16, expand=64):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        with tf.variable_scope('fire_module'):
            x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
            x = Activation('selu')(x)

            left = Convolution2D(expand, (1, 1), padding='valid')(x)
            left = Activation('selu')(left)

            right = Convolution2D(expand, (3, 3), padding='same')(x)
            right = Activation('selu')(right)

            x = concatenate([left, right], axis=channel_axis)
        return x

    include_top = True
    input_tensor = None
    input_shape = (128, 128, 3)
    pooling = None
    classes = y_test.shape[-1]

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=227,
        min_size=48,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)])}},
        strides=(2, 2),
        padding='valid',
        activation='selu',
    )(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    nth = 1
    n_layer = conditional({{choice([1, 2, 3, 4])}})
    for _ in range(n_layer):
        squeeze = nth * {{choice([8, 16])}}
        expand = squeeze * {{choice([3, 4])}}
        x = fire_module(x, squeeze=squeeze, expand=expand)
        x = fire_module(x, squeeze=squeeze, expand=expand)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        nth += 1

    n_layer = conditional({{choice([1, 2, 3])}})
    for _ in range(n_layer):
        squeeze = nth * {{choice([8, 16])}}
        expand = squeeze * {{choice([3, 4])}}
        x = fire_module(x, squeeze=squeeze, expand=expand)
        x = fire_module(x, squeeze=squeeze, expand=expand)

        nth += 1

    # It's not obvious where to cut the network...
    # Could do the 8th or 9th layer... some work recommends cutting earlier layers.

    x = Dropout({{uniform(0, 1)}})(x)

    x = Convolution2D(
        classes,
        (1, 1),
        padding='valid',
        activation='selu',
    )(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of .
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size={{choice([32, 64, 128])}},
        epochs=100,
        verbose=2,
        validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def justdeep(x_train, y_train, x_test, y_test):
    include_top = True
    input_tensor = None
    input_shape = (128, 128, 3)
    pooling = None
    classes = y_test.shape[-1]
    relu = 'elu'

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=227,
        min_size=48,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5)])}},
        strides=(2, 2),
        padding='valid',
        activation=relu,
    )(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    nth = 1
    #n_layer = conditional({{choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}})
    n_layer = conditional({{choice([1, 2, 3, 4, 5, 6])}})
    for _ in range(n_layer):
        squeeze = nth * {{choice([8, 16])}}
        expand = squeeze * {{choice([3, 4])}}
        x = fire_module(x, squeeze=squeeze, expand=expand)
        x = fire_module(x, squeeze=squeeze, expand=expand)
        #x = Dropout({{uniform(0, 1)}})(x)

        nth += 1

    # It's not obvious where to cut the network...
    # Could do the 8th or 9th layer... some work recommends cutting earlier layers.

    x = Dropout({{uniform(0, 1)}})(x)

    x = Convolution2D(
        classes,
        (1, 1),
        padding='valid',
        activation=relu,
    )(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of .
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

    model.fit(
        x_train, y_train, batch_size=64, epochs=100, verbose=2, validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


def mydeep(x_train, y_train, x_test, y_test):
    from keras.applications.imagenet_utils import _obtain_input_shape
    from keras import backend as K
    from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
    from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
    from keras.models import Model
    from keras.engine.topology import get_source_inputs
    from keras.utils import get_file
    from keras.utils import layer_utils

    def fire_module(x, squeeze=16, expand=64):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        with tf.variable_scope('fire_module'):
            x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
            x = Activation('elu')(x)

            left = Convolution2D(expand, (1, 1), padding='valid')(x)
            left = Activation('elu')(left)

            right = Convolution2D(expand, (3, 3), padding='same')(x)
            right = Activation('elu')(right)

            x = concatenate([left, right], axis=channel_axis)
        return x

    include_top = True
    input_tensor = None
    input_shape = (128, 128, 3)
    pooling = None
    classes = y_test.shape[-1]

    input_shape = _obtain_input_shape(
        input_shape,
        default_size=227,
        min_size=48,
        data_format=K.image_data_format(),
        require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(
        {{choice([16, 32, 64])}},
        {{choice([(3, 3), (5, 5), (7, 7), (9, 9), (11, 11)])}},
        strides=(2, 2),
        padding='valid',
        activation='elu',
    )(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    nth = 1
    n_layer = conditional({{choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}})
    for _ in range(n_layer):
        squeeze = nth * {{choice([8, 16])}}
        expand = squeeze * {{choice([3, 4])}}
        x = fire_module(x, squeeze=squeeze, expand=expand)
        x = fire_module(x, squeeze=squeeze, expand=expand)

        nth += 1

    # It's not obvious where to cut the network...
    # Could do the 8th or 9th layer... some work recommends cutting earlier layers.

    x = Dropout({{uniform(0, 1)}})(x)

    x = Convolution2D(
        classes,
        (1, 1),
        padding='valid',
        activation='elu',
    )(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of .
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
        metrics=['accuracy'])

    model.fit(
        x_train,
        y_train,
        batch_size={{choice([32, 64, 128])}},
        epochs=100,
        verbose=2,
        validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}
