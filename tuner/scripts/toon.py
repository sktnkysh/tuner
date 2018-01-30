#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import subprocess
from datetime import datetime
import argparse

import tuner
from tuner import utils
from tuner import load_data
from tuner import augment_data
from tuner import use_hyperas
from tuner import net
from tuner.dataset import ClassificationDataset, AugmentDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', nargs='?', help='dataset directory')
    parser.add_argument(
        '-o',
        '--output',
        dest='save_model_file',
        type=str,
        default='models/tuner.{}.model.hdf5'.format(int(datetime.now().timestamp())),
        help='dataset directory')
    parser.add_argument('-b', '--batchsize', dest='bs', type=int, default=32, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='epochs')
    parser.add_argument('-n', '--only-net', dest='is_only_net', action='store_true', help='epochs')
    args = parser.parse_args()

    if args.src_dir:
        src_dir = args.src_dir
    elif not sys.stdin.isatty():
        src_dir = sys.stdin.read().rstrip()
    else:
        parser.print_help()

    ### 
    brain = ClassificationDataset(src_dir)

    if args.is_only_net:
        pass
    else:
        brain = AugmentDataset(brain)

        ### Search best condition of data augmentation
        brain.search_opt_augment(model=net.neoaug)

        ### Execute data augmentation with best condition
        brain.augment_dataset()

    ### Tuning hyper parameter of CNN
    best_condition, best_model = use_hyperas.exec_hyperas(
        brain.augmented_dir,
        brain.validation_dir,
        net.simplenet,
        batch_size=args.bs,
        epochs=args.epochs,
        optimizer='adam',
        rescale=1. / 255)
    best_model.save(args.save_model_file)
    print('saved best model', args.save_model_file)
    shutil.rmtree('standard_datasets')
