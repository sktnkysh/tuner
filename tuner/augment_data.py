import os
import json
import Augmentor
from tuner import utils


def _augment_dir(source_dir, sampling_size=10):
    with open('cond.json', 'r') as f:
        conds = json.load(f)
    p = Augmentor.Pipeline(source_dir)
    p.flip_left_right(probability=0.5)
    if conds['conditional']:
        p.crop_random(probability=1, percentage_area=0.8)
        p.resize(probability=1, width=96, height=96)
    if conds['conditional_1']:
        p.random_erasing(probability=0.5, rectangle_area=0.2)
    if conds['conditional_2']:
        p.shear(probability=0.3, max_shear_left=2, max_shear_right=2)
    p.sample(sampling_size)


def augment_dir(src_dir, output_dir, sampling_size=10):
    #if os.path.exists(output_dir):
    #    raise 'exsists {}'.format(output_dir)
    _augment_dir(src_dir, sampling_size)
    utils.mvtree(os.path.join(src_dir, 'output'), output_dir)


def augment_dataset(src_dir, output_dir, sampling_size=10):
    labels = os.listdir(src_dir)
    utils.mkdir(output_dir)
    for label in labels:
        read_dir = os.path.join(src_dir, label)
        write_dir = os.path.join(output_dir, label)
        #utils.mkdir(write_dir)
        augment_dir(read_dir, write_dir)
