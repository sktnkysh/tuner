# coding: utf-8

# In[2]:

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

from tuner import utils
from tuner.tune import fit, fit_generator

from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform, conditional

# In[3]:

train_dir = 'dataset/brain/train'
validation = 'dataset/brain/validation'
df_train = utils.df_fromdir_classed(train_dir)
df_test = utils.df_fromdir_classed(train_dir)

x_train, y_train = load_data.load_fromdf(df_train)
x_test, y_test = load_data.load_fromdf(df_test)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# In[4]:

import Augmentor

# In[6]:


#@compute_loss(epochs=200)
def tuning_aug(n_out, input_shape=(96, 96, 3)):
    batch_size = 32
    epochs = 200
    steps_per_epoch = 100

    #p = Augmentor.Pipeline()
    #p.flip_left_right(probability=0.5)
    #p.crop_random(probability=1, percentage_area=0.8)
    #p.resize(probability=1, width=96, height=96)
    #p.status()
    #g = p.keras_generator_from_array(x_train, y_train, batch_size=batch_size)
    #g = ((x / 255., y) for (x, y) in g)

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

    #model.fit_generator(
    #    g,
    #    steps_per_epoch=steps_per_epoch,
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
    )
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


# In[10]:


def data():
    return x_train, y_train, x_test, y_test


best_run, best_model = optim.minimize(
    model=tuning_aug, data=data, algo=tpe.suggest, max_evals=10, trials=Trials())
x_train, y_train, x_test, y_test = data()
# print("#####################################")
print("Evalutation of best performing model:")
score = best_model.evaluate(x_test, y_test)
print('loss:', score[0])
print('acc:', score[1])
print("Best performing model chosen hyper-parameters:")
best_model.summary()
print(best_run)
