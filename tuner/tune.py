import tensorflow as tf
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Conv2D, Flatten
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings, GlobalAveragePooling2D
from keras.models import Sequential, Model
from keras.utils import np_utils, to_categorical

from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, conditional


def compute_loss(lossfun='categorical_crossentropy',
                 optimizer=keras.optimizers.rmsprop(lr=0.005, decay=1e-6),
                 metrics=['accuracy'],
                 batch_size=32,
                 epochs=1):

    def wrap(target_net):

        def forward(x_train, y_train, x_test, y_test):
            n_out = y_train.shape[-1]
            input_shape = x_train.shape[1:]
            model = target_net(n_out=n_out)
            model.compile(loss=lossfun, optimizer=optimizer, metrics=metrics)
            model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(x_test, y_test))
            score, acc = model.evaluate(x_test, y_test, verbose=0)
            print('Test accuracy:', acc)
            return {'loss': -acc, 'status': STATUS_OK, 'model': model}

        return forward

    return wrap
