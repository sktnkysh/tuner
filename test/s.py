
# coding: utf-8

# In[29]:


import os
import numpy as np
from PIL import Image


# In[32]:


import keras
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


# In[31]:


import data
import utils
import net


# In[3]:


#model = load_model('models/doctor.brain.model.h5')
#model = load_model('models/model_best.hdf5')
model = load_model('models/model72.hdf5')

