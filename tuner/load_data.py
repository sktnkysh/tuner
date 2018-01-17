import os
import shutil
import itertools

import numpy as np
from numpy.random import choice
import pandas as pd
from PIL import Image

from tuner import utils

RESIZE = 28


def get_labels_fromdir(dataset_dir):
    # hyperas don't allowed itertor
    #can = next(os.walk(dataset_dir))[1]

    can = os.listdir(dataset_dir)
    can = [c for c in can if os.path.isdir(os.path.join(dataset_dir, c))]
    return [c for c in can if c != 'output']


def df_fromdir(classed_dir, columns=['name', 'label']):
    fname_label = []

    labels = get_labels_fromdir(classed_dir)
    for label in labels:
        for fname in os.listdir(os.path.join(classed_dir, label)):
            d = (fname, label)
            fname_label.append(d)
    df = pd.DataFrame(fname_label, columns=columns)
    df['handle'] = utils.path_join(df['label'], df['name'])
    df['path'] = utils.path_join(classed_dir, df['label'], df['name'])
    return df


def df_fromdir_eyes(eyes_dir, teaching_file='label.csv'):
    f = os.path.join(eyes_dir, teaching_file)
    ends = f.split('.')[-1]
    sep = {'csv': ',', 'tsv': '\t'}[ends]
    df = pd.read_csv(f, sep=sep)

    df_can = pd.DataFrame(os.listdir(eyes_dir), columns=['name'])
    df_can = df_can[df_can['name'].str.match('.+\.(jpg|png|gif|JPG|JPEG)')]
    df_can['id'] = df_can['name'].apply(lambda s: ''.join(s.split('.')[:-1]))

    df = pd.merge(df, df_can, on='id', how='left')

    df['path'] = utils.path_join(eyes_dir, df['name'])
    df['handle'] = utils.path_join(df['label'], df['name'])
    return df


def df_fromdir_brain(brain_dir):

    # PD061.jpg -> [PD, 061, .jpg]
    def strint_separator(f):
        return [''.join(it) for _, it in itertools.groupby(f, str.isdigit)]

    def extra_label(f):
        return strint_separator(f)[0]

    df = pd.DataFrame(os.listdir(brain_dir), columns=['name'])
    df = df[df['name'].str.match('.+\.(jpg|png|gif|JPG|JPEG)')]

    df['label'] = df['name'].map(extra_label)
    df['path'] = utils.path_join(brain_dir, df['name'])
    df['handle'] = utils.path_join(df['label'], df['name'])
    return df


def classed_dir_fromdf(src_df, dst_dir, val_size=0.1):
    df = src_df

    labels = list(set(df['label']))

    utils.mkdir(dst_dir)
    for label in labels:
        utils.mkdir(os.path.join(dst_dir, label))

    df['dst_path'] = utils.path_join(dst_dir, df['label'], df['name'])

    for k, col in df.iterrows():
        shutil.copy(str(col['path']), str(col['dst_path']))


def ready_dir_fromdf(src_df, dst_dir, val_size=0.1):
    df = src_df
    train_dir = os.path.join(dst_dir, 'train')
    val_dir = os.path.join(dst_dir, 'validation')

    labels = list(set(df['label']))

    utils.mkdir(dst_dir)
    utils.mkdir(train_dir)
    utils.mkdir(val_dir)
    for label in labels:
        utils.mkdir(os.path.join(train_dir, label))
        utils.mkdir(os.path.join(val_dir, label))

    df_train, df_val = train_val_split_df(df, val_size=val_size)
    df_train['dst_path'] = utils.path_join(train_dir, df['label'], df['name'])
    df_val['dst_path'] = utils.path_join(val_dir, df['label'], df['name'])

    for k, col in df_train.iterrows():
        shutil.copy(str(col['path']), str(col['dst_path']))

    for k, col in df_val.iterrows():
        shutil.copy(str(col['path']), str(col['dst_path']))


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
    resize = None if resize is None else\
            resize if type(resize) is tuple else\
            (resize, resize)

    img = Image.open(f)
    if not resize is None:
        img = img.resize(resize, Image.LANCZOS)
    img = img.convert('RGB')
    return np.asarray(img) * rescale
