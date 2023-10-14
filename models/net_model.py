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
    
    def __init__(self,  n_layers: int = 5, n_neurons: int = 256, learning_rate: float = 0.01,
                input_shape: tuple = (BOARD_WIDTH, BOARD_HEIGHT, 3)) -> None:
        """Generate a dense neural network with n_layers hidden layers and 256 neurons per layer.

        :param n_layers: Number of Hidden layers, defaults to 5
        :type n_layers: int, optional
        :param n_neurons: Number of neurons by layer, defaults to 256
        :type n_neurons: int, optional
        :param learning_rate: Learning rate of the optimizer, defaults to 0.01
        :type learning_rate: float, optional
        :param input_shape: Shape of the board, defaults to (BOARD_WIDTH, BOARD_HEIGHT, 3)
        :type input_shape: tuple, optional
        """                   
        self.model = Sequential()
        # Input layer
        self.model.add(Flatten(input_shape=input_shape))
        # Hidden layers
        for _ in range(n_layers - 1):
            self.model.add(Dense(n_neurons, activation='relu',
                                  kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        # Output layer
        self.model.add(Dense(5, activation='softmax'))
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.model.compile(optimizer=optimizer,
                            loss="mean_squared_error",
                            metrics=[tf.keras.metrics.RootMeanSquaredError()])

class ConvulutionNet():

    def __init__(self, n_conv_layers: int = 6, n_dense_layers: int = 3, n_neurons: int = 256,
                learning_rate: float = 0.01, input_shape: tuple = (BOARD_WIDTH, BOARD_HEIGHT, 3)) -> None:
        """Generate a convolutional neural network with n_layers hidden layers and 256 neurons per layer.

        :param n_conv_layers: Number of convolutional layers, defaults to 5
        :type n_conv_layers: int, optional
        :param n_dense_layers: Number of dense layers, defaults to 3
        :type n_dense_layers: int, optional
        :param n_neurons: Number of neurons by Dense layer, defaults to 256
        :type n_neurons: int, optional
        :param learning_rate: Learning rate of the optimizer, defaults to 0.01
        :type learning_rate: float, optional
        :param input_shape: Shape of the board, defaults to (BOARD_WIDTH, BOARD_HEIGHT, 3)
        :type input_shape: tuple, optional
        """

        self.model = Sequential()
        for _ in range(n_conv_layers //2 ):
            self.model.add(Conv2D(32, (4, 4), activation='relu', padding='same',
                                   input_shape = input_shape,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            
        self.model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

        for _ in range(n_conv_layers - n_conv_layers //2):
            self.model.add(Conv2D(32, (4, 4), activation='relu', padding='same',
                                   input_shape = input_shape,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        
        self.model.add(Flatten())

        # Dense layers
        for _ in range(n_dense_layers - 1):
            self.model.add(Dense(n_neurons, activation='relu',
                                  kernel_regularizer = tf.keras.regularizers.l2(0.01)))
        
        # Output layer
        self.model.add(Dense(5, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error",
                            metrics=[tf.keras.metrics.RootMeanSquaredError()])
