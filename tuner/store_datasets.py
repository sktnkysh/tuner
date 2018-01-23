
# coding: utf-8

# In[1]:

import os
import shutil
import pandas as pd
import sqlite3

import utils

from database import db, mongodb


# In[2]:

dbname = 'curontab.db'
conn = sqlite3.connect(dbname)
c = conn.cursor()
c.execute('PRAGMA foreign_keys=true')


# In[3]:

client = pymongo.MongoClient('localhost',27017)
db = client.curontab
co = db.dataset


# In[4]:

try:
    c.execute('drop table model')
    c.execute('drop table upload_data')
except:
    pass

q = '''
create table model (
    id integer primary key,
    name varcher(64),
    validation_table varcher(64),
    dataset_id varcher(32),
    dataset_name varcher(64)
)
'''
c.execute(q)

q = '''
create table upload_data (
    id integer primary key,
    name varcher(64),
    path varcher(64)
)
'''
c.execute(q)
conn.commit()

co.remove({})


# In[5]:

dataset_name = 'eyes'
dataset_dir = os.path.join('dataset',dataset_name)
labels = list(set(os.listdir(dataset_dir)))
labels.sort()
id2label = {i:label for i,label in enumerate(labels) }
label2id = {label:i for i,label in enumerate(labels) }


# In[6]:

dataset_info = {
    'labels': list(labels),
    'label2id': label2id,
    'name': dataset_name,
    'data':[],
    'models':[],
}
dataset_info


# In[7]:

dataset_id = co.insert(dataset_info)
dataset_id


# In[8]:

data_table = 'data_{}'.format(dataset_id)
data_df = load_data.df_fromdir_classed(dataset_dir)
data_df['path'] = utils.format_dirname(dataset_dir) + data_df['label'] + '/' + data_df['fname']
data_df.to_sql(data_table, conn)
co.update({'_id':dataset_id},{'$set':{'data':data_table}})


# In[9]:

dataset_name = 'brain'
dataset_dir = os.path.join('dataset',dataset_name)
labels = set(os.listdir(dataset_dir))
id2label = {i:label for i,label in enumerate(labels) }
label2id = {label:i for i,label in enumerate(labels) }

dataset_info = {
    'labels': list(labels),
    'label2id': label2id,
    'name': dataset_name,
    'data':[],
    'models':[],
}
dataset_id = co.insert(dataset_info)
data_table = 'data_{}'.format(dataset_id)
data_df = load_data.df_fromdir_classed(dataset_dir)
data_df['path'] = utils.format_dirname(dataset_dir) + data_df['label'] + '/' + data_df['fname']
data_df.to_sql(data_table, conn)
co.update({'_id':dataset_id},{'$set':{'data':data_table}})


# In[10]:

conn.commit()
conn.close()

