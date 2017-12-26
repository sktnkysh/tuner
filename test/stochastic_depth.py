
# coding: utf-8

# In[1]:


import os
import utils
import data
import numpy as np
from PIL import Image


# In[3]:


orig = data.orig()


# In[5]:


x_train, y_train, x_test, y_test = orig


# In[6]:


def get_p_survival(block=0, nb_total_blocks=110, p_survival_end=0.5, mode='linear_decay'):
    """
    See eq. (4) in stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf
    """
    if mode == 'uniform':
        return p_survival_end
    elif mode == 'linear_decay':
        return 1 - ((block + 1) / nb_total_blocks) * (1 - p_survival_end)
    else:
        raise


# In[7]:


def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)


# In[8]:


def stochastic_survival(y, p_survival=1.0):
    # binomial random variable
    survival = K.random_binomial((1,), p=p_survival)
    # during testing phase:
    # - scale y (see eq. (6))
    # - p_survival effectively becomes 1 for all layers (no layer dropout)
    return K.in_test_phase(tf.constant(p_survival, dtype='float32') * y, 
                           survival * y)


# In[33]:


def stochastic_depth_residual_block(x, nb_filters=16, block=0, nb_total_blocks=110, subsample_factor=1):
    """
    Stochastic depth paper: http://arxiv.org/pdf/1603.09382v1.pdf
    
    Residual block consisting of:
    - Conv - BN - ReLU - Conv - BN
    - identity shortcut connection
    - merge Conv path with shortcut path
    Original paper (http://arxiv.org/pdf/1512.03385v1.pdf) then has ReLU,
    but we leave this out: see https://github.com/gcr/torch-residual-networks
    Additional variants explored in http://arxiv.org/pdf/1603.05027v1.pdf
    
    some code adapted from https://github.com/dblN/stochastic_depth_keras
    """
    
    prev_nb_channels = K.int_shape(x)[3]

    if subsample_factor > 1:
        subsample = (subsample_factor, subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        shortcut = AveragePooling2D(pool_size=subsample, dim_ordering='tf')(x)
        if nb_filters > prev_nb_channels:
            shortcut = Lambda(zero_pad_channels,
                              arguments={'pad': nb_filters - prev_nb_channels})(shortcut)
    else:
        subsample = (1, 1)
        # shortcut: identity
        shortcut = x

    y = Convolution2D(nb_filters, 3, 3, subsample=subsample,
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    y = BatchNormalization(axis=3)(y)
    y = Activation('relu')(y)
    y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1),
                      init='he_normal', border_mode='same', dim_ordering='tf')(y)
    y = BatchNormalization(axis=3)(y)
    
    p_survival = get_p_survival(block=block, nb_total_blocks=nb_total_blocks, p_survival_end=0.5, mode='linear_decay')
    y = Lambda(stochastic_survival, arguments={'p_survival': p_survival})(y)
    
    out = Add()([y, shortcut])

    return out


# In[53]:





# In[62]:


nb_classes = y_test.shape[-1]

img_rows, img_cols = 128, 128
img_channels = 3
batch_size = 32
blocks_per_group = 33

nb_epoch = 1


# In[63]:


import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, AveragePooling2D, BatchNormalization
from keras.layers import Add, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import keras.backend as K

inputs = Input(shape=(img_rows, img_cols, img_channels))

x = Convolution2D(16, 3, 3, 
                  init='he_normal', border_mode='same', dim_ordering='tf')(inputs)
x = BatchNormalization(axis=3)(x)
x = Activation('relu')(x)

for i in range(0, blocks_per_group):
    nb_filters = 16
    x = stochastic_depth_residual_block(x, nb_filters=nb_filters, 
                                        block=i, nb_total_blocks=3 * blocks_per_group, 
                                        subsample_factor=1)

for i in range(0, blocks_per_group):
    nb_filters = 32
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = stochastic_depth_residual_block(x, nb_filters=nb_filters, 
                                        block=blocks_per_group + i, nb_total_blocks=3 * blocks_per_group, 
                                        subsample_factor=subsample_factor)

for i in range(0, blocks_per_group):
    nb_filters = 64
    if i == 0:
        subsample_factor = 2
    else:
        subsample_factor = 1
    x = stochastic_depth_residual_block(x, nb_filters=nb_filters, 
                                        block=2 * blocks_per_group + i, nb_total_blocks=3 * blocks_per_group, 
                                        subsample_factor=subsample_factor)

x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid', dim_ordering='tf')(x)
x = Flatten()(x)

predictions = Dense(nb_classes, activation='softmax')(x)


# In[64]:


model = Model(input=inputs, output=predictions)
model.summary()


# In[65]:


model.compile(optimizer=SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[66]:


from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping

# Learning rate schedule
def lr_sch(epoch):
    if epoch < nb_epoch * 0.5:
        return 0.1
    elif epoch < nb_epoch * 0.75:
        return 0.01
    else:
        return 0.001
lr_scheduler = LearningRateScheduler(lr_sch)


# In[67]:


import Augmentor
p = Augmentor.Pipeline()
p.skew_left_right(probability=0.05, magnitude=0.1)
p.skew_top_bottom(probability=0.05, magnitude=0.1)
p.skew_tilt(probability=0.05, magnitude=0.1)
p.skew(probability=0.05, magnitude=0.1)
p.shear(probability=0.05, max_shear_left=1, max_shear_right=1)
g = p.keras_generator_from_array(x_train, y_train, batch_size=batch_size)


# In[ ]:


history = model.fit_generator(
    g,
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // batch_size,
    shuffle=True,
    epochs=nb_epoch,
    verbose=1,
    callbacks=[lr_scheduler]
)

