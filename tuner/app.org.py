# coding: utf-8

# In[2]:

import os
import random

from urllib.parse import urlparse

from PIL import Image
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, jsonify, make_response
from flask_cors import CORS, cross_origin

from werkzeug import secure_filename

import utils
import crud
import pred

from database import db, mongodb

# In[3]:

app = Flask(__name__, static_folder='dataset')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif', 'JPG'])
try:
    app.config['ROOT_DIR'] = os.path.dirname(os.path.abspath(__file__))
except:
    app.config['ROOT_DIR'] = './'

# In[3]:

tester = app.test_client()
render = lambda url: print(tester.get(url).data.decode('utf-8'))

# In[4]:


@app.route('/')
def index():
    return render_template('index.html')


# In[5]:


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


# In[6]:


@app.route('/v1/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("file[]")
    res = []
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            data_id = crud.upload_data(filename, file_path)
            r = crud.uploaded_data(data_id)
            res.append(r)
    return make_response(jsonify(res))


# In[7]:


@app.route('/v1/uploads')
def res_uploaded_files():
    limit = request.args.get('limit', default=-1, type=int)
    df = crud.uploads_df()
    res = list(df.T.to_dict().values())[:limit]
    return make_response(jsonify(res))


# In[8]:


@app.route('/uploads/<filename>')
def res_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# In[9]:


@app.route('/v1/scores')
def res_Scores():
    #model_id = request.args.get('model', type=str)
    #try:
    #    int(model_id)
    #except:
    #    model_id = None
    model_id = request.args.get('model', type=int)
    limit = request.args.get('limit', default=-1, type=int)
    preds = crud.scores(model_id)
    res = []
    for pred in preds[:limit]:
        res.append({
            'data_id': int(pred['data_id']),
            'fname': pred['fname'],
            'label': pred['label'],
            'match': pred['match'],
            'label_pred': pred['pred_label'],
            'pred': int(pred['pred']),
            'y_pred': [float(e) for e in pred['y_pred']]
        })
    return make_response(jsonify(res))


#render('/v1/scores?model=1')

# In[10]:


@app.route('/v1/preds')
def res_pred():
    model_id = request.args.get('model', type=int)
    fname = request.args.get('fname', type=str)
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    print(data_path, model_id)
    preds = pred.judge_img_by_data_path(data_path, model_id)
    res = {
        'model_id': preds['model_id'],
        'path': preds['path'],
        'label_pred': str(preds['label_pred']),
        'pred': int(preds['pred']),
        'y_pred': [float(e) for e in preds['y_pred']]
    }
    return make_response(jsonify(res))


#render('/v1/preds?fname=105_0018.JPG&model=1')

# In[11]:


@app.route('/v1/datasets')
def res_datasets():
    name = request.args.get('name', type=str)
    res = crud.dataset_by_name(name) if name else crud.datasets()
    return make_response(jsonify(res))


#render('/v1/datasets')

# In[12]:


@app.route('/v1/datasets/<string:model_id>')
def res_dataset(model_id):
    res = crud.dataset_by_id(model_id)
    return make_response(jsonify(res))


#render('/v1/datasets/5a2524cf94a27008a9abc554')

#render('/v1/datasets/5a2524cf94a27008a9abc554')

# In[13]:


@app.route('/v1/models')
def res_models():
    limit = request.args.get('limit', default=-1, type=int)
    df = crud.models()
    res = list(df.T.to_dict().values())[:limit]
    return make_response(jsonify(res))


#render('/v1/models')

# In[14]:


@app.route('/v1/models/<int:model_id>')
def res_model(model_id):
    res = crud.model_by_id(model_id)
    return make_response(jsonify(res))


#render('/v1/models/1')

# In[ ]:

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='eval model')
    parser.add_argument('--port', '-p', type=int, default=3000, help='port')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='gpu On or Off')
    args = parser.parse_args()

    app.debug = True
    app.run(port=args.port)
