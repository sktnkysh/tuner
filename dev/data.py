import numpy as np
import pandas
import utils
import Augmentor

from sklearn.model_selection import train_test_split


def orig_raw():
    train_dname = '../data/orig/train'
    test_dname = '../data/orig/validation'
    #x_train, y_train = utils.load_data_fromdir(train_dname)
    x_train, y_train = utils.load_data_fromdir(train_dname)
    x_test, y_test = utils.load_data_fromdir(test_dname)

    return x_train, y_train, x_test, y_test


def brain_raw():
    train_dname = '../data/brain/train'
    test_dname = '../data/brain/validation'
    x_train, y_train = utils.oversampling_fromdir(train_dname, 80)
    x_test, y_test = utils.load_data_fromdir(test_dname)

    return x_train, y_train, x_test, y_test


def dbrain():
    train_dname = '../dataset/brain/train'
    test_dname = '../dataset/brain/validation'
    x_train, y_train = utils.oversampling_fromdir(train_dname, 80)
    x_test, y_test = utils.load_data_fromdir(test_dname)

    return x_train, y_train, x_test, y_test


def eyes_raw():
    train_dname = '../data/brain/train'
    test_dname = '../data/brain/validation'
    x_train, y_train = utils.oversampling_fromdir(train_dname, 400)
    x_test, y_test = utils.load_data_fromdir(test_dname)

    return x_train, y_train, x_test, y_test
    """
    data_dir = '../micin/eyes/'

    df = utils.load_eyes_data('../micin/ynzw_result.csv.tsv')
    df = utils.oversampling(df, 400)
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    x_data, y_data, categories = utils.load_from_dataframe(
        df, dir_name=data_dir)

    x_test, y_test = x_data[:len(x_data) // 10], y_data[:len(y_data) // 10]
    x_train, y_train = x_data[len(x_data) // 10:], y_data[len(y_data) // 10:]

    return x_train, y_train, x_test, y_test
    """


def braineyes_raw():
    x_brain = brain_raw()[0]
    x_eyes = eyes_raw()[0]
    n = min(x_brain.shape[0], x_eyes.shape[0])
    x_brain = x_brain[:n]
    x_eyes = x_eyes[:n]

    data = []
    for img in x_brain:
        data.append((img, np.array([0, 1])))
    for img in x_eyes:
        data.append((img, np.array([1, 0])))
    np.random.shuffle(data)

    x_data = []
    y_data = []
    for d in data:
        x_data.append(d[0])
        y_data.append(d[1])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_test, y_test = x_data[:len(x_data) // 10], y_data[:len(y_data) // 10]
    x_train, y_train = x_data[len(x_data) // 10:], y_data[len(y_data) // 10:]
    return x_train, y_train, x_test, y_test


def eyes_aug():
    x_train, y_train, x_test, y_test = eyes_raw()

    p = augmentor.pipeline()
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    #p.skew_left_right(probability=0.05, magnitude=0.1)
    #p.skew_top_bottom(probability=0.05, magnitude=0.1)
    #p.skew_tilt(probability=0.05, magnitude=0.1)
    #p.skew(probability=0.05, magnitude=0.1)
    #p.shear(probability=0.05, max_shear_left=1, max_shear_right=1)
    g = p.keras_generator_from_array(x_train, y_train, batch_size=1)

    n = len(x_train) * 4

    x_train = []
    y_train = []
    for _ in range(n):
        x, y = next(g)
        x_train.append(x[0])
        y_train.append(y[0])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train, x_test, y_test


def orig():
    train_dname = '../data/orig/train'
    test_dname = '../data/orig/validation'
    x_train, y_train = utils.oversampling_fromdir(train_dname, 80)
    #x_train, y_train = utils.load_data_fromdir(train_dname)
    x_test, y_test = utils.load_data_fromdir(test_dname)

    return x_train, y_train, x_test, y_test
    x_train, y_train, x_test, y_test = orig_raw()
    n = len(x_train) * 2

    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    g = p.keras_generator_from_array(x_train, y_train, batch_size=n)
    x_train, y_train = next(g)

    return x_train, y_train, x_test, y_test


def brain():
    x_train, y_train, x_test, y_test = brain_raw()
    n = len(x_train) * 2

    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    #p.flip_top_bottom(probability=0.5)
    g = p.keras_generator_from_array(x_train, y_train, batch_size=n)
    x_train, y_train = next(g)

    return x_train, y_train, x_test, y_test


def eyes():
    x_train, y_train, x_test, y_test = eyes_raw()
    n = len(x_train) * 2

    p = Augmentor.Pipeline()
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    g = p.keras_generator_from_array(x_train, y_train, batch_size=n)
    x_train, y_train = next(g)

    return x_train, y_train, x_test, y_test


def _eyes():
    from keras.preprocessing.image import ImageDataGenerator

    data_dir = '../micin/eyes/'

    df = utils.load_eyes_data('../micin/ynzw_result.csv.tsv')
    df = utils.oversampling(df, 400)
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True)
    x_data, y_data, categories = utils.load_from_dataframe(
        df, dir_name=data_dir)

    x_test, y_test = x_data[:len(x_data) // 10], y_data[:len(y_data) // 10]
    x_train, y_train = x_data[len(x_data) // 10:], y_data[len(y_data) // 10:]

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
    )
    x_train, y_train = utils.data_augmentation(
        x_train, y_train, datagen=datagen)

    return x_train, y_train, x_test, y_test
