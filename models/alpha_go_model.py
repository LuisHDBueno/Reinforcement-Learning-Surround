import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../game'))

import surround as s

BOARD_HEIGHT = s.BOARD_HEIGHT
BOARD_WIDTH = s.BOARD_WIDTH

class DenseNet():
    
    def __init__(self, input_shape: tuple = (BOARD_WIDTH, BOARD_HEIGHT, 3), n_layers: int = 5) -> None:
        self.model = Sequential()
        self.model.add(Flatten(input_shape=input_shape))
        for _ in range(n_layers):
            self.model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Dense(1, activation='linear'))
        num_epochs = 25
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.05, decay_steps=num_epochs, decay_rate=0.025, staircase=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])

class ConvulutionNet():

    def __init__(self, input_shape: tuple = (BOARD_WIDTH, BOARD_HEIGHT, 3)) -> None:
        self.model = Sequential()
        self.model.add(Conv2D(32, (4, 4), activation='relu', padding='same', input_shape = input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Conv2D(32, (4, 4), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Conv2D(32, (4, 4), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
        self.model.add(Conv2D(16, (2, 2), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Conv2D(16, (2, 2), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Conv2D(16, (2, 2), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        self.model.add(Dense(1, activation='linear'))
        num_epochs = 25
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.05, decay_steps=num_epochs, decay_rate=0.025, staircase=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])


    


"""
model = Sequential()
model.add(Conv2D(32, (4, 4), activation='relu', padding='same', input_shape=(1, BOARD_WIDTH, BOARD_HEIGHT), kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(32, (4, 4), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(32, (4, 4), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(16, (2, 2), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(16, (2, 2), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Conv2D(16, (2, 2), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(Dense(1, activation='linear'))
num_epochs = 25
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.05, decay_steps=num_epochs, decay_rate=0.025, staircase=False)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])"""