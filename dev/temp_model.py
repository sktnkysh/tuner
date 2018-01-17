#coding=utf-8

try:
    import os
except:
    pass

try:
    import subprocess
except:
    pass

try:
    import json
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    from hyperopt import hp
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform, conditional
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    import keras
except:
    pass

try:
    from keras import backend as K
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.layers import BatchNormalization, Flatten, Dense
except:
    pass

try:
    from keras.layers import Input, Conv2D, Convolution2D
except:
    pass

try:
    from keras.layers import Activation, concatenate, Dropout
except:
    pass

try:
    from keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
except:
    pass

try:
    from keras.models import Sequential, Model
except:
    pass

try:
    from keras.utils import np_utils, to_categorical
except:
    pass

try:
    from keras.applications.imagenet_utils import _obtain_input_shape
except:
    pass

try:
    import Augmentor
except:
    pass

try:
    from tuner import utils
except:
    pass

try:
    from tuner import load_data
except:
    pass

try:
    from tuner import net
except:
    pass

try:
    from hyperas_data import data
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

train_dir = 'standard_datasets/5a5f131db037cb8223459336/train' 
test_dir = 'standard_datasets/5a5f131db037cb8223459336/validation'
resize = 96 
rescale = 1 
df = load_data.df_fromdir(train_dir)
x_train, y_train = load_data.load_fromdf(df, resize=resize, rescale=rescale)
df = load_data.df_fromdir(test_dir)
x_test, y_test = load_data.load_fromdf(df, resize=resize, rescale=rescale)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



def keras_fmin_fnct(space):

    params = eval(open('tmp_params.json', 'r').read())

    n_out = params['n_out']
    input_shape = tuple(params['input_shape'])
    batch_size = params['batch_size']
    epochs = params['epochs']
    lossfun = params['lossfun']
    optimizer = params['optimizer']
    metrics = ['accuracy']

    steps_per_epoch = len(x_train) // batch_size

    p = Augmentor.Pipeline()

    p.flip_left_right(probability=0.5)
    if conditional(space['conditional']):
        p.crop_random(probability=1, percentage_area=0.8)
        p.resize(probability=1, width=96, height=96)
    if conditional(space['conditional_1']):
        p.random_erasing(probability=0.5, rectangle_area=0.2)
    if conditional(space['conditional_2']):
        p.shear(probability=0.3, max_shear_left=2, max_shear_right=2)
    print('-' * 80)
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

def get_space():
    return {
        'conditional': hp.choice('conditional', [True, False]),
        'conditional_1': hp.choice('conditional_1', [True, False]),
        'conditional_2': hp.choice('conditional_2', [True, False]),
    }
