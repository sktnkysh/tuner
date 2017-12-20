
# coding: utf-8

# In[1]:

import os
from urllib.parse import urlparse
import sqlite3
import pymongo


# MONGO_URI = 'mongodb://heroku_kf4kn4jq:g9ronedfpaanu9u849ka78k0h1@ds129966.mlab.com:29966/heroku_kf4kn4jq'

# In[2]:

db_sqlite = 'curontab.db'


# In[3]:

MONGO_URI = os.environ.get('MONGODB_URI')
if MONGO_URI:
    dbname = urlparse(MONGO_URI).path[1:]
    client = pymongo.MongoClient(MONGO_URI)
    db = client[dbname]
else:
    client = pymongo.MongoClient('localhost', 27017)
    db = client.curontab
mongodb = db

