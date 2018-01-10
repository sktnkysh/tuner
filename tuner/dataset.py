import os
import json
from bson.objectid import ObjectId

import pandas as pd

import Augmentor

import tuner
from tuner import utils
from tuner import load_data
from tuner import augment_data
from tuner import tune_cnn
from tuner import use_hyperas
from tuner import net


class ClassificationDataset(object):

    def __init__(self, classed_dataset_dir):
        self._id = ObjectId()
        self.id = str(self._id)
        self.original_dataset_path = classed_dataset_dir 
        self.path = 'standard_datasets/{}'.format(self.id)
        self.train_dir = os.path.join(self.path, 'train')
        self.validation_dir = os.path.join(self.path, 'validation')

        utils.mkdir(self.path)
        load_data.format_dataset(self.original_dataset_path, self.path, mode='eyes')

        self.df_train = load_data.df_fromdir(self.train_dir)
        self.df_validation = load_data.df_fromdir(self.validation_dir)

        df1 = self.df_train
        df2 = self.df_validation
        df1['t/v'] = 'train'
        df2['t/v'] = 'validatoin'
        self.df = pd.concat([df1, df2])

        self.n_label = self.n_labels = len(self.df_train['label'].drop_duplicates())

    def counts_train_data(self):
        return self.df_train['label'].value_counts().to_dict()

    def counts_validation_data(self):
        return self.df_validation['label'].value_counts().to_dict()

    def _load_train_data(self, resize=28, rescale=1):
        self.resize = resize
        self.rescale = rescale
        df = load_data.df_fromdir(self.train_dir)
        x_train, y_train = load_data.load_fromdf(df, resize=self.resize, rescale=self.rescale)
        self.x_train = x_train
        self.y_train = y_train
        self.train_data = (x_train, y_train)
        return x_train, y_train

    def _load_validation_data(self, resize=28, rescale=1):
        self.resize = resize
        self.rescale = rescale
        df = load_data.df_fromdir(self.validation_dir)
        x_val, y_val = load_data.load_fromdf(df, resize=self.resize, rescale=self.rescale)
        self.x_validation = self.x_val = x_val
        self.y_validation = self.y_val = y_val
        self.validation_data = (x_val, y_val)
        return x_val, y_val

    def load_data(self, resize=28, rescale=1):
        self.resize = resize
        self.rescale = rescale
        x_train, y_train = self._load_train_data(self.resize, self.rescale)
        x_val, y_val = self._load_validation_data(self.resize, self.rescale)
        return x_train, x_val, y_train, y_val


class AugmentDataset(object):

    def __init__(self, standard_dataset):
        self.dataset = standard_dataset
        self.df_validation = self.dataset.df_validation
        self.augment_condition = 'cond.json'
        self.augmented_dir = os.path.join(self.dataset.path, 'auged')
        self.train_dir = self.augmented_dir
        self.p = Augmentor.Pipeline(self.dataset.train_dir)

    def search_opt_augment(self, model=net.aug):
        best_condition, best_model = use_hyperas.exec_hyperas(\
            self.dataset.train_dir,
            self.dataset.validation_dir, model)
        with open(self.augment_condition, 'w') as f:
            json.dump(best_condition, f)

    def augment_dataset_custom_p(self, sampling_size=None):
        sampling_size =\
            sampling_size if sampling_size else\
            min(self.dataset.counts_train_data().values()) * 4
        augment_data.augment_dataset_custom_p(
            self.dataset.train_dir, self.augmented_dir, sampling_size=sampling_size, p=self.p)
        print('augment dataset done.')
        self.df_augmented = load_data.df_fromdir(self.augmented_dir)
        self.df_train = self.df_augmented

    def augment_dataset(self, sampling_size=None):
        sampling_size =\
            sampling_size if sampling_size else\
            min(self.dataset.counts_train_data().values()) * 4
        augment_data.augment_dataset(
            self.dataset.train_dir,
            self.augmented_dir,
            condition_file=self.augment_condition,
            sampling_size=sampling_size,
            p=None)
        self.df_augmented = load_data.df_fromdir(self.augmented_dir)
        self.df_train = self.df_augmented

    def _load_augmented_data(self, resize=28, rescale=1):
        self.resize = resize
        self.rescale = rescale
        df = load_data.df_fromdir(self.augmented_dir)
        x_train, y_train = load_data.load_fromdf(df, resize=self.resize, rescale=self.rescale)
        return x_train, y_train

    def load_data(self, resize=28, rescale=1):
        self.resize = resize
        self.rescale = rescale
        x_train, y_train = self._load_augmented_data(self.resize, self.rescale)
        x_val, y_val = self.dataset._load_validation_data(self.resize, self.rescale)
        self.x_train = x_train
        self.y_train = y_train
        self.x_validation = self.x_val = x_val
        self.y_validation = self.y_val = y_val
        self.train_data = (x_train, y_train)
        self.validation_data = (x_val, y_val)
        return x_train, x_val, y_train, y_val

    def search_opt_cnn(self, model=net.simplenet):
        best_condition, best_model = use_hyperas.exec_hyperas(\
            self.dataset.train_dir,
            self.dataset.validation_dir, model)
        fname = 'simplenet.hdf5'
        best_model.save(fname)
        return fname