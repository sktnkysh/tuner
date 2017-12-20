
# coding: utf-8

# In[1]:

import os
import shutil
import random
import pandas as pd
import sqlite3

from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix

from PIL import Image
import Augmentor

import utils
import crud
import net

from database import db, mongodb


# In[2]:

imgnet = net.lenet1
dataset_name = 'brain'
batch_size = 32
epochs = 10


# In[ ]:

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--net', '-n', type=str,
                    help='model file name')
parser.add_argument('--weights', '-w', type=str,
                    help='weights file name')
parser.add_argument('--data', '-d', type=str,
                    help='set_data file name')
parser.add_argument('--id', '-i', type=str, default=None,
                    help='')
parser.add_argument('--epochs', '-e', type=int, default=100,
                    help='n epochs')
parser.add_argument('--batch', '-b', type=int, default=32,
                    help='batch size')
parser.add_argument('--augment', '-a', type=str, default=None,
                    help='brain or eyes')
args = parser.parse_args()

imgnet = eval(args.net)
dataset_name = args.data
batch_size = args.batch
epochs = args.epochs


# In[3]:

dbname = 'curontab.db'
conn = sqlite3.connect(dbname)
c = conn.cursor()
c.execute('PRAGMA foreign_keys=true')


# In[4]:

client = pymongo.MongoClient('localhost',27017)
db = client.curontab
co = db.dataset


# DBからデータセットの属性を取得

# In[5]:

crud.datasets()


# In[6]:

crud.models()


# In[7]:

dataset= dataset_info = crud.dataset_by_name(dataset_name)
dataset_id = dataset['id']
labels = dataset['labels']
labels.sort()
label2id = dataset['label2id']
id2label = {i:label for label,i in label2id.items()}
df = data_df = crud.data_df_by_dataset_name(dataset_name)

ROOT_DIR = '.'
dataset_dir = os.path.join(ROOT_DIR, 'dataset', dataset['name'])


# In[8]:

df_train, df_test = utils.train_test_split_df(df)


# In[9]:

if dataset_name == 'eyes':
    df_train = utils.oversampling_df(df_train, 600)
x_train, y_train = utils.data_fromdf(df_train, dataset_dir, label2id, resize=(96,96))
x_test, y_test = utils.data_fromdf(df_test, dataset_dir, label2id, resize=(96,96))


# In[10]:

p = Augmentor.Pipeline()

if dataset_name == 'eyes':
    p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
    p.crop_random(probability=0.2, percentage_area=0.95)
    p.resize(probability=1, width=96, height=96)
if dataset_name == 'brain':
    p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
    p.crop_random(probability=0.2, percentage_area=0.95)
    p.resize(probability=1, width=96, height=96)

g = p.keras_generator_from_array(x_train, y_train, batch_size=batch_size)
g = ( (xs/255,ys) for (xs,ys) in g)


# In[11]:

import keras
n_out = len(labels)
model = net.lenet1(n_out)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])


# In[12]:

model.fit_generator(
    g,
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // batch_size,
    epochs=epochs,
    shuffle=True,
    verbose=1,
    )


# In[13]:

df_preds = pd.DataFrame(model.predict(x_test))
f = pd.concat([df_preds,df_test], axis=1)
f = f.rename(columns={'index':'data_id'})
f['true'] = [ label2id[label] for label in f['label']]
f['pred'] = y_pred = model.predict(x_test).argmax(axis=1)
f['pred_label'] = [id2label[e] for e in f['pred']]
f['match'] = f['true'] == f['pred']
f['t_or_v'] = 'validation'

f


# In[14]:

from bson.objectid import ObjectId
validation_table = 'validation_{}'.format(str(ObjectId()))
f.to_sql(validation_table, conn)


# In[15]:

model_name = '{}_{}'.format(dataset_name, imgnet.__name__)
q = '''
insert into model(
    name,
    validation_table,
    dataset_id,
    dataset_name
) values(?,?,?,?)
'''
v = (model_name, validation_table, dataset_id, dataset_name)
c.execute(q,v)


# In[28]:

model.save('models/{}.hdf5'.format(model_name))


# In[29]:

model_id = c.lastrowid
co = db.dataset
co.update({'name':dataset_name},{'$push':{
    'models':model_id}})


# In[30]:

conn.commit()
conn.close()

