import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import Augmentor
import data

import snet
from snet import sixsqueeze

import argparse
parser = argparse.ArgumentParser(description='xception doctor')
parser.add_argument('--net', '-n', type=str,
                    help='model file name')
parser.add_argument('--weights', '-w', type=str,
                    help='weights file name')
parser.add_argument('--data', '-d', type=str,
                    help='set_data file name')
parser.add_argument('--id', '-i', type=str, default=None,
                    help='')
parser.add_argument('--epochs', '-e', type=int, default=200,
                    help='n epochs')
parser.add_argument('--batch', '-b', type=int, default=32,
                    help='batch size')
parser.add_argument('--augment', '-a', type=str, default=None,
                    help='brain or eyes')
args = parser.parse_args()


model = eval(args.net+'(classes=4)')
if args.weights:
    model.load_weights(args.weights)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])
model.summary()


batch_size = args.batch
epochs = args.epochs

x_train, y_train, x_test, y_test = eval(args.data + '()')
p = Augmentor.Pipeline()
p.flip_left_right(probability=0.5)
p.flip_top_bottom(probability=0.5)
if args.augment == 'erase':
    p.random_erasing(probability=0.5, rectangle_area=0.2)
if args.augment == 'brain':
    p.random_distortion(probability=0.2, grid_width=2,
                        grid_height=2, magnitude=2)
    p.random_erasing(probability=0.5, rectangle_area=0.5)
if args.augment == 'eyes':
    p.skew_left_right(probability=0.1, magnitude=0.1)
    p.skew_top_bottom(probability=0.1, magnitude=0.1)
    p.skew_tilt(probability=0.1, magnitude=0.1)
    p.skew(probability=0.1, magnitude=0.1)
    p.shear(probability=0.1, max_shear_left=1, max_shear_right=1)
    p.random_erasing(probability=0.5, rectangle_area=0.2)
if args.augment == 'weak':
    p.skew_left_right(probability=0.05, magnitude=0.1)
    p.skew_top_bottom(probability=0.05, magnitude=0.1)
    p.skew_tilt(probability=0.05, magnitude=0.1)
    p.skew(probability=0.05, magnitude=0.1)
    p.shear(probability=0.05, max_shear_left=1, max_shear_right=1)
g = p.keras_generator_from_array(x_train, y_train, batch_size=batch_size)


history = model.fit_generator(
    g,
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // batch_size,
    shuffle=True,
    epochs=epochs,
    verbose=1,
)


# save model weights


def fname_parse(fname):
    fname = fname.split('/')[-1]
    attrs = [a for a in fname.split('.') if a.find('=') != -1]
    d = {attr.split('=')[0]: attr.split('=')[1] for attr in attrs}
    return d


def random_str(n):
    import string
    import random
    return ''.join(
        [random.choice(string.ascii_letters + string.digits) for i in range(n)])


import os

result_dir = './models/'
idx = args.id if args.id else random_str(8)

fname = 'id={}.{}.model.h5'.format(
        idx, '.'.join(args.net.split('/')[-1].split('.')[:-2]))
model.save(os.path.join(result_dir, fname))
print(fname, 'saved')

fname = 'id={}.{}.weights.h5'.format(
        idx, '.'.join(args.net.split('/')[-1].split('.')[:-2]))
model.save_weights(os.path.join(result_dir, fname))
print(fname, 'saved')


