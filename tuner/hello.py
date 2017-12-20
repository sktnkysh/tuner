# hello.py
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

import tensorflow as tf
import multiprocessing as mp


app = Flask(__name__, static_folder='dataset')
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif', 'JPG'])
try:
    app.config['ROOT_DIR'] = os.path.dirname(os.path.abspath(__file__))
except:
    app.config['ROOT_DIR'] = './'



@app.route('/')
def index():
    return render_template('index.html')




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']




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



@app.route('/v1/uploads')
def res_uploaded_files():
    limit = request.args.get('limit', default=-1, type=int)
    df = crud.uploads_df()
    res = list(df.T.to_dict().values())[:limit]
    return make_response(jsonify(res))




@app.route('/uploads/<filename>')
def res_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



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

if __name__ == '__main__':
    app.run()
