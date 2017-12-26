import os
import shutil
import itertools

import numpy as np
from numpy.random import choice
import pandas as pd
from PIL import Image

from tuner import utils

RESIZE = 28


def df_fromdir(data_dir, columns=['name', 'label']):
    fname_label = []

    labels = os.listdir(data_dir)
    for label in labels:
        for fname in os.listdir(os.path.join(data_dir, label)):
            d = (fname, label)
            fname_label.append(d)
    df = pd.DataFrame(fname_label, columns=columns)
    df['path'] = utils.format_dirname(data_dir + '/' + df['label'] + '/' + df['name'])
    return df


def df_fromdir_brain(dir_name):

    def strint_separator(f):
        return [''.join(it) for _, it in itertools.groupby(f, str.isdigit)]

    fname_label = []
    # for removing .DS_Store
    flist = [f for f in os.listdir(dir_name) if '.jpg' in f]
    for fname in flist:
        label = strint_separator(fname)[0]  # 'N','MS','PD' or'PS'
        d = (fname, label)
        fname_label.append(d)

    df = pd.DataFrame(fname_label, columns=['fname', 'label'])
    return df


def _format_brain(src_dir, dst_dir):

    df = df_fromdir_brain(src_dir)
    labels = list(set(df['label']))

    utils.mkdir(dst_dir)
    for label in labels:
        utils.mkdir(os.path.join(dst_dir, label))

    for k, col in df.iterrows():
        read_fname = os.path.join(src_dir, col['fname'])
        write_fname = os.path.join(dst_dir, col['label'], col['fname'])
        shutil.copy(read_fname, write_fname)


def format_brain(src_dir, dst_dir, val_size=0.1):
    train_dir = os.path.join(dst_dir, 'train')
    val_dir = os.path.join(dst_dir, 'validation')

    df = df_fromdir_brain(src_dir)
    labels = list(set(df['label']))

    utils.mkdir(dst_dir)
    utils.mkdir(train_dir)
    utils.mkdir(val_dir)
    for label in labels:
        utils.mkdir(os.path.join(train_dir, label))
        utils.mkdir(os.path.join(val_dir, label))

    df_train, df_val = train_val_split_df(df, val_size=val_size)
    for k, col in df_train.iterrows():
        read_fname = os.path.join(src_dir, col['fname'])
        write_fname = os.path.join(train_dir, col['label'], col['fname'])
        shutil.copy(read_fname, write_fname)

    for k, col in df_val.iterrows():
        read_fname = os.path.join(src_dir, col['fname'])
        write_fname = os.path.join(val_dir, col['label'], col['fname'])
        shutil.copy(read_fname, write_fname)


def train_val_split_df(data_frame, val_size=0.1):

    df = data_frame
    labels = set(df['label'])
    n = min(df['label'].value_counts())
    sampling_size = int(n * val_size)
    sampling_idx = np.array([
        choice(df[df.label == label].index, sampling_size, replace=False) for label in labels
    ]).flatten()
    val_df = df.loc[sampling_idx].reset_index(drop=True)
    train_df = df.drop(sampling_idx).reset_index(drop=True)
    return train_df, val_df


def load_fromdf(dataframe, label2id=None, resize=RESIZE, rescale=1):

    df = dataframe
    labels = list(set(df['label']))
    l2i = label2id if label2id else {label: i for i, label in enumerate(labels)}

    x_data = []
    y_data = []
    for idx, row in df.iterrows():
        y = l2i[row['label']]
        f = row['path']

        x = arr_fromf(f, resize=resize, rescale=rescale)
        x_data.append(x)
        y_data.append(y)
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data


def load_fromdir(dataset_dir, label2id=None, resize=RESIZE, rescale=1):
    df = df_fromdir(dataset_dir)
    x_data, y_data = load_fromdf(df, label2id=label2id, resize=resize, rescale=rescale)
    return x_data, y_data


def arr2img(arr):
    return Image.fromarray(np.uint8(arr))


def arr_fromf(f, resize=RESIZE, rescale=1):
    resize = resize if type(resize) is tuple else (resize, resize)
    img = Image.open(f).resize(resize, Image.LANCZOS)
    if f.endswith('png') or f.endswith('PNG'):
        img = img.convert('RGB')
    return np.asarray(img) * rescale
