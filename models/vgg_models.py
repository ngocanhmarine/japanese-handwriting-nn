"""VGG-type models"""
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import *
from keras.layers.normalization import *
from keras.optimizers import *
import numpy as np


def M7_1(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Convolution2D(
        64, 3, strides=1, padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(192, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def M13(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()
    model.add(Convolution2D(
        32, 3, strides=1, padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, strides=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(256, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(512, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def M6_1(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()
    model.add(Convolution2D(
        32, 3, 3, padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def M9(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Convolution2D(
        64, 3, strides=1, padding='valid', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, strides=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512, 3, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)
    return model


def M11(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Convolution2D(
        64, 3, strides=1, padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))

    if weights_path:
        model.load_weights(weights_path)
    return model


def M6_2(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Convolution2D(
        32, 3, strides=1, padding='valid', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, strides=1, padding='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def M6_3(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Convolution2D(
        64, 3, strides=1, padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def M7_1(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Convolution2D(
        64, 3, strides=1, padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3,))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, strides=1,))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def M12(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()

    model.add(Convolution2D(
        64, 3, strides=1, padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, strides=1))
    model.add(Activation('relu'))
    model.add(Convolution2D(512, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def M8(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()
    model.add(Convolution2D(
        32, 3, strides=1, padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 3, strides=1, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def VGG_vis(weights_path=None, input_shape=(1, 64, 64), n_output=None):
    model = Sequential()
    model.add(Convolution2D(
        64, 11, strides=11, padding='same', input_shape=input_shape, data_format='channels_first'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, 7, strides=7, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(256, 5, strides=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(512, 3, strides=1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_output))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    if weights_path:
        model.load_weights(weights_path)
    return model


def M16(weights_path='weights/keras_VGG-16-katakana_weights.h5',
        input_shape=(1, 64, 64),
        n_output=None,
        freeze_layers=False):

    if freeze_layers:
        trainable = False
    else:
        trainable = True

    model = Sequential()
    model.add(Convolution2D(64, 3, strides=1, trainable=trainable,
                            name='conv1_1', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, strides=1, activation='relu', name='conv1_2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, strides=1, activation='relu', name='conv2_1'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, strides=1, activation='relu', name='conv2_2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, strides=1, activation='relu', name='conv3_1'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, strides=1, activation='relu', name='conv3_2'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, strides=1, activation='relu', name='conv3_3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv4_1'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv4_2'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv4_3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv5_1'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv5_2'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv5_3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_output))
    model.add(Activation('softmax'))

    if weights_path:
        try:
            model.load_weights(weights_path)
        except:
            print("Can't load weights!")

    return model


def M16_drop(input_shape=(1, 64, 64),
             n_output=None,
             freeze_layers=False):

    if freeze_layers:
        trainable = False
    else:
        trainable = True

    model = Sequential()
    model.add(Convolution2D(64, 3, strides=1, trainable=trainable,
                            name='conv1_1', input_shape=input_shape, data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, strides=1, activation='relu', name='conv1_2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, strides=1, activation='relu', name='conv2_1'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, strides=1, activation='relu', name='conv2_2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, strides=1, activation='relu', name='conv3_1'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, strides=1, activation='relu', name='conv3_2'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, strides=1, activation='relu', name='conv3_3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv4_1'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv4_2'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv4_3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv5_1'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv5_2'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, strides=1, activation='relu', name='conv5_3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_output))
    model.add(Activation('softmax'))

    return model
