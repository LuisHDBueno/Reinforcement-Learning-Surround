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

class NeuralNet():
    def predict(self, board):
        """Predict the reward distribution for each action.

        :param board: Current game board
        :type board: np.array
        :return: Array containing the relative reward for each action, interpreted as a probability distribution
        :rtype: np.array
        """         
        return self.model.predict(board)
        
    def train(self, boards, rewards, batch_size=64, epochs=1):
        """Train the model with the given boards and rewards.

        :param boards: Array of boards from Monte Carlo Tree Search
        :type boards: np.array
        :param rewards: Array of rewards from Monte Carlo Tree Search
        :type rewards: np.array
        :param batch_size: Size of the batch to train the model
        :type batch_size: int
        :param epochs: Number of trainment epochs, defaults to 1
        :type epochs: int, optional
        """
        self.model.fit(boards, rewards, batch_size=batch_size, epochs=epochs, verbose=0)

    def play(self, board):
        """Choose the best action to play.

        :param board: Current game board
        :type board: np.array
        :return: Best action to play
        :rtype: int
        """            
        best_action = np.argmax(self.predict(board))
        return best_action + 1
        
    def save(self, file_name):
        """Save the model.

        :param file_name: Name of the file to save the model
        :type file_name: str
        """            
        self.model.save(f'./saved_models/{file_name}.keras')

    def load(self, file_name):
        """Load pre-trained model.

        :param file_name: Name of the file to load the model
        :type file_name: str
        """            
        self.model = tf.keras.models.load_model(f'./saved_models/{file_name}.keras')


class DenseNet(NeuralNet):
    
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
        self.model.add(Dense(4, activation='softmax'))
        
        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.model.compile(optimizer=optimizer,
                            loss="mean_squared_error",
                            metrics=[tf.keras.metrics.RootMeanSquaredError()])

class ConvolutionNet(NeuralNet):

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
        for _ in range(n_conv_layers // 2):
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
        self.model.add(Dense(4, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.model.compile(optimizer=optimizer, loss="mean_squared_error",
                            metrics=[tf.keras.metrics.RootMeanSquaredError()])
