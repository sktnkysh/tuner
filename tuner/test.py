
# coding: utf-8

# In[1]:

from app import app


# In[2]:

tester = app.test_client()
render = lambda url: print(tester.get(url).data.decode('utf-8'))


# ### /uploads
# 静的ファイル，アップロードされたファイルが格納される <br/>
# <a href='/uploads/101_0015.JPG'>/uploads/101_0015.JPG</a>

# ### /dataset
# 静的ファイル，モデルの訓練，検証に用いたデータセットが格納されている

# ### /v1/upload
# 画像をアップロードする

# In[3]:

render('/v1/upload')


# ### /v1/scores
# モデルのvalidationを表示する <br/>
# パラメータ'model'は必須，modelのidを表す

# In[4]:

render('/v1/scores?model=1&limit=3')


# ### /v1/preds
# アップロードされた画像に対して予測結果を返す <br/> 
# パラメータfnameとmodelは必須

# In[5]:

render('/v1/preds?fname=eyes_AB_105_0018.JPG&model=1')


# ### /v1/uploads
# アップロードされたファイル一覧

# In[6]:

render('/v1/uploads?limit=3')


# ### /v1/datasets

# In[7]:

render('/v1/datasets')


# In[8]:

render('/v1/datasets?name=eyes')


# ### /v1/models

# In[9]:

render('/v1/models')


# In[10]:

render('/v1/models/4')


# ### /v1

# ### /v1
