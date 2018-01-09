import os
import json
from bson.objectid import ObjectId

import tuner
from tuner import utils
from tuner import load_data
from tuner import augment_data
from tuner import tune_cnn
from tuner import use_hyperas
from tuner import net


class ClassificationDataset(object):

    def __init__(self, dataset_dir):
        self._id = ObjectId()
        self.id = str(self._id)
        self.size = 96
        self.scale = 1.
        self.original_dataset_path = dataset_dir
        self.path = 'standard_datasets/{}'.format(self.id)
        self.train_dir = os.path.join(self.path, 'train')
        self.validation_dir = os.path.join(self.path, 'validation')

        utils.mkdir(self.path)
        load_data.format_dataset(self.original_dataset_path, self.path, mode='eyes')

        self.df_train = load_data.df_fromdir(self.train_dir)
        self.df_validation = load_data.df_fromdir(self.validation_dir)
        self.n_label = self.n_labels = len(self.df_train['label'].drop_duplicates())

    def counts_train_data(self):
        return self.df_train['label'].value_counts().to_dict()

    def counts_validation_data(self):
        return self.df_validation['label'].value_counts().to_dict()

    def load_train_data(self):
        df = load_data.df_fromdir(self.train_dir)
        x_train, y_train = load_data.load_fromdf(df, resize=self.size, rescale=self.scale)
        self.x_train = x_train
        self.y_train = y_train
        self.train_data = (x_train, y_train)
        return x_train, y_train

    def load_validation_data(self):
        df = load_data.df_fromdir(self.validation_dir)
        x_val, y_val = load_data.load_fromdf(df, resize=self.size, rescale=self.scale)
        self.x_validation = self.x_val = x_val
        self.y_validation = self.y_val = y_val
        self.validation_data = (x_val, y_val)
        return x_val, y_val


class AugmentDataset(object):

    def __init__(self, standard_dataset):
        self._id = ObjectId()
        self.id = str(self._id)
        self.dataset = standard_dataset
        self.augment_condition = 'cond.json'

    def search_opt_augment(self, model=net.aug):
        best_condition, best_model = use_hyperas.exec_hyperas(\
            self.dataset.train_dir,
            self.dataset.validation_dir, model)
        with open(self.augment_condition, 'w') as f:
            json.dump(best_condition, f)

    def augment_dataset(self, sampling_size=None):
        self.augmented_dir = os.path.join(self.dataset.path, 'auged')
        sampling_size =\
            sampling_size if sampling_size else\
            min(self.dataset.counts_train_data().values()) * 4
        augment_data.augment_dataset(
            self.dataset.train_dir,
            self.augmented_dir,
            condition_file=self.augment_condition,
            sampling_size=sampling_size)

    def search_opt_cnn(self, model=net.simplenet):
        best_condition, best_model = use_hyperas.exec_hyperas(\
            self.dataset.train_dir,
            self.dataset.validation_dir, model)
        fname = 'simplenet.hdf5'
        best_model.save(fname)
        return fname
