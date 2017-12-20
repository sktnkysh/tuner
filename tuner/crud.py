
# coding: utf-8

# In[2]:

import os
import shutil
import random
import pandas as pd
import sqlite3
from bson.objectid import ObjectId

import utils

from database import db, mongodb, db_sqlite


# In[3]:

def format_doc(doc):
    doc['id'] = str(doc['_id'])
    doc.pop('_id')
    return doc


# In[4]:

def datasets():
    co = db.dataset
    docs = co.find()
    res = []
    for doc in docs:
        res.append(format_doc(doc))
    return res
datasets()


# In[5]:

def dataset_by_id(dataset_id):
    co = db.dataset
    doc = co.find_one({'_id':ObjectId(dataset_id)})
    doc = format_doc(doc)
    doc['models'] = list(set(doc['models']))
    return doc
#dataset_by_id('5a2496c6b037cbff714fc2f5')


# In[6]:

def dataset_by_name(dataset_name):
    co = db.dataset
    doc = co.find_one({'name':dataset_name})
    doc = format_doc(doc)
    doc['models'] = list(set(doc['models']))
    return doc
dataset_by_name('eyes')


# In[7]:

def data_df_by_dataset_id(dataset_id):
    with sqlite3.connect(db_sqlite) as con:
        q = '''
        select * from data_{}
        '''.format(dataset_id)
        df = pd.io.sql.read_sql(q, con)
    return df
#data_df_by_dataset_id('5a2496c6b037cbff714fc2f5')


# In[8]:

def data_df_by_dataset_name(dataset_name):
    data = dataset_by_name(dataset_name)['data']
    with sqlite3.connect(db_sqlite) as con:
        q = '''
        select * from {}
        '''.format(data)
        df = pd.io.sql.read_sql(q, con)
    return df
#data_df_by_dataset_name('eyes')


# In[9]:

def models():
    with sqlite3.connect(db_sqlite) as con:
        q = '''
        select * from model
        '''
        df = pd.io.sql.read_sql(q,con)
    return df
models()


# In[10]:

def model_by_id(id):
    id = int(id)
    with sqlite3.connect(db_sqlite) as con:
        q = '''
        select * from model where id={}
        '''.format(id)
        df = pd.io.sql.read_sql(q,con)
        model_info = df.T.to_dict()[0]
    return model_info
#model_by_id(1)


# In[11]:

def validation_df(key):
    key = str(key)
    with sqlite3.connect(db_sqlite) as con:
        q = '''
        select * from {}
        '''.format(key)
        df = pd.io.sql.read_sql(q,con)
    return df
#validation_table('validation_5a249051b037cbfa20b6e074')


# In[12]:

def scores(model_id):
    model_info = model_by_id(model_id)
    df = validation_df(model_info['validation_table'])
    #dataset_info = dataset_by_id(model_info['dataset_id'])
    dataset_name = model_info['dataset_name']
    dataset_info = dataset_by_name(dataset_name)

    n_label = len(dataset_info['labels'])
    columns = ['data_id', 'fname', 'path', 'label', 'true', 'pred', 'pred_label', 'match']
    ds = list(df[columns].T.to_dict().values())
    
    y_preds = df[[str(i) for i in range(n_label)]].as_matrix().tolist()
    
    for d, y_pred in zip(ds, y_preds):
        d.update({
            'accuracy': df['match'].sum()/df['match'].count(),
            'y_pred': y_pred
        })
    res = ds
    return res
#scores(2)


# In[13]:

def upload_data(data_name, data_path):
    with sqlite3.connect(db_sqlite) as con:
        c = con.cursor()
        q = '''
        insert into upload_data(name, path) values(?,?)
        '''
        v = (data_name, data_path)
        c.execute(q,v)
        data_id = c.lastrowid
        con.commit()
    return data_id

#upload_data('test2', 'test/path2')


# In[4]:

def uploads_df():
    with sqlite3.connect(db_sqlite) as con:
        q = '''
        select * from upload_data
        '''
        df = pd.io.sql.read_sql(q,con)
    return df
uploads_df()


# In[15]:

def uploaded_data(data_id):
    with sqlite3.connect(db_sqlite) as con:
        q = '''
        select * from upload_data where id={}
        '''.format(data_id)
        df = pd.io.sql.read_sql(q,con)
        res = list(df.T.to_dict().values())[0]
    return res
try:
    uploaded_data(1)
except:
    pass

