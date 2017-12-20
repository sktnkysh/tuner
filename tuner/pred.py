
# coding: utf-8

# In[1]:

import os
import shutil
import random
import pandas as pd

import numpy as np

import sqlite3

from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix

from keras.models import Model, load_model
from keras.layers import Conv2D, Flatten, Dense, GlobalAveragePooling2D

from PIL import Image
import Augmentor

import utils
import crud


# In[2]:

model_df = crud.models()
model_df['path'] = ['models/{}.hdf5'.format(name) for name in model_df['name']]
model_df


# In[3]:

def try_load_model(model_path):
    try:
        return load_model(model_path)
    except:
        return None


# In[5]:

models = {}

for itr in model_df.iterrows():
    model_id = itr[1]['id']
    model = try_load_model(itr[1]['path'])
    if model:
        models.update({
            model_id: model
        })


# In[6]:



# with sqlite3.connect('curontab.db') as con:
#     c = con.cursor()
#     q = '''
#     update model set dataset_name = 'eyes' where dataset_id= "5a24a8bab037cb1164d683c2"
#     '''#.format(name='brain')
#     c.execute(q)
#     con.commit()    

# with sqlite3.connect('curontab.db') as con:
#     c = con.cursor()
#     q = '''
#     alter table model add column dataset_name[varcher(64)]
#     '''
#     c.execute(q)
#     con.commit()
#     

# In[7]:

def judge_img_by_data_path(data_path, model_id):
    model_info = crud.model_by_id(model_id)
    dataset_name = model_info['dataset_name']
    label2id = crud.dataset_by_name(dataset_name)['label2id']
    id2label = {id:label for label, id in label2id.items()}

    model = models[model_id]
    batch, h, w, c = model.input_shape
    x = utils.img2arr(data_path, resize=(h,w), rescale=1./255)
    xs = np.array([x])

    y_preds = model.predict(xs)
    y_pred = y_preds[0]
    pred = y_pred.argmax()
    label_pred = id2label[pred]
    
    res = {
        'y_pred': y_pred.tolist(),
        'model_id': model_id,
        #'dataset_name': dataset_name,
        #'label2id': label2id,
        #'id2label': id2label,
        'path': data_path,
        'label_pred': id2label[pred],
        'pred': pred,
    }
    return res
#judge_img_by_data_path('dataset/eyes/N/148_0015.JPG', 1)


# In[8]:

def judge_imgs_by_data_path(data_paths, model_id):
    model_info = crud.model_by_id(model_id)
    dataset_name = model_info['dataset_name']
    label2id = crud.dataset_by_name(dataset_name)['label2id']
    id2label = {id:label for label, id in label2id.items()}

    model = models[model_id]
    batch, h, w, c = model.input_shape
    xs = np.array([utils.img2arr(f, resize=(h,w), rescale=1./255) for f in data_paths])

    y_preds = model.predict(xs)
    df = pd.DataFrame()
    df['pred'] = y_preds.argmax(axis=1)
    df['label_pred'] = [ id2label[pred] for pred in df['pred']]
    df['path'] = data_paths
    df['model_id'] = model_id
    ds = list(df.T.to_dict().values())
    ds

    for d, y_pred in zip(ds, y_preds.tolist()):
        d.update({'y_pred':y_pred})
    res = ds
    return res
imgs = ['dataset/eyes/N/148_0015.JPG', 'dataset/eyes/AB/159_0003.JPG']
#judge_imgs_by_data_path(imgs, 2)


# In[10]:

def judge_img_by_data_id(data_id, model_id):
    data_path = crud.uploaded_data(data_id)['path']
    res = judge_img_by_data_path(data_path, model_id)
    return res
#judge_img_by_data_id(43, 4)


# In[12]:

def judge_imgs_by_data_id(data_ids, model_id):
    df = crud.uploads_df()
    data_paths = df[df['id'].isin(data_ids)]['path'].values.tolist()
    res = judge_imgs_by_data_path(data_paths, model_id)
    return res
#judge_imgs_by_data_id([27,31,35], 4)

