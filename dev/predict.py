
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


# In[6]:


validation_dir = '../data/orig/validation'
labels = os.listdir(validation_dir)
labels.sort()
labels


# In[27]:


idx2label = {i:label for i, label in enumerate(labels)}


# In[34]:


for iii, label in enumerate(labels):
    
    n_ok = 0
    flist = os.listdir(os.path.join(validation_dir, label))
    for f in flist:
        fname = os.path.join(validation_dir, label, f)
        x = utils.img2arr(fname, rescale=(96,96))/255.
        xs = np.expand_dims(x, axis=0)
        y = model.predict(xs).argmax(axis=1)[0]
        
        if y==iii:
            n_ok+=1
        print(fname, '\t',idx2label[y])
        
    acc = n_ok/len(flist)
    print('accuracy:',acc)
    print()

