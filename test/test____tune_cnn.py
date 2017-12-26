
# coding: utf-8

# In[1]:


from tuner.tune_cnn import search_model


# In[2]:


def data():
    from keras.utils import to_categorical
    from tuner.load_data import load_fromdir
    dataset_dir = 'original_dataset/brain'
    resize=96
    x_train, y_train = load_fromdir(os.path.join(dataset_dir, 'auged'), resize=resize)
    x_test, y_test = load_fromdir(os.path.join(dataset_dir, 'validation'), resize=resize)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test


# In[3]:


search_model(data)

