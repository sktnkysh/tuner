
# coding: utf-8

# In[1]:


import os
import shutil
import tuner
from tuner.load_data import format_brain, df_fromdir, train_val_split_df, load_fromdir, load_fromdf
from tuner.augment_data import search_condition, augment_dataset


# In[5]:


dataset_dir = 'original_dataset/brain'


# In[6]:


format_brain('../examples/micin-dataset/brain', dataset_dir)


# In[7]:


df = df_fromdir('brain')

df_train, df_val = train_val_split_df(df)
df_train.shape, df_val.shape


# In[8]:


df['label'].value_counts()


# In[9]:


aug_res_file = 'cond.json'


# In[10]:


def data():
    from keras.utils import to_categorical
    from tuner.load_data import load_fromdir
    dataset_dir = 'original_dataset/brain'
    resize=96
    x_train, y_train = load_fromdir(os.path.join(dataset_dir, 'train'), resize=resize)
    x_test, y_test = load_fromdir(os.path.join(dataset_dir, 'validation'), resize=resize)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test

search_condition(data, aug_res_file)


# In[11]:


out_dir=os.path.join(dataset_dir, 'auged')
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

augment_dataset(
    src_dir=os.path.join(dataset_dir, 'train'),
    out_dir=out_dir,
    condition_file=aug_res_file,
    sampling_size=200)

