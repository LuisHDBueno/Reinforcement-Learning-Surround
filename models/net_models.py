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
    def predict(self, board: np.array) -> np.array:
        """Predict the reward distribution for each action.

        :param board: Current game board
        :type board: np.array
        :return: Array containing the relative reward for each action, interpreted as a probability distribution
        :rtype: np.array
        """         
        return self.model.predict(board)
        
    def fit(self, boards: np.array, rewards: np.array, batch_size: int = 64, epochs: int = 1) -> None:
        """Fit the model with the given boards and rewards.

        :param boards: Array of boards from Monte Carlo Tree Search
        :type boards: np.array
        :param rewards: Array of rewards from Monte Carlo Tree Search
        :type rewards: np.array
        :param batch_size: Size of the batch to fit the model
        :type batch_size: int
        :param epochs: Number of trainment epochs, defaults to 1
        :type epochs: int, optional
        """
        self.model.fit(boards, rewards, batch_size=batch_size, epochs=epochs, verbose=0)

    def play(self, board: np.array) -> int:
        """Choose the best action to play.

        :param board: Current game board
        :type board: np.array
        :return: Best action to play
        :rtype: int
        """            
        best_action = np.argmax(self.predict(board))
        return best_action + 1
    
    def get_win_rate(self, adversary: 'NeuralNet', num_games: int = 50) -> float:
        """Get the win rate of the model against a fixed adversary.

        :param adversary: Neural network to play against
        :type adversary: NeuralNet
        :return: Win rate of the model against the adversary
        :rtype: float
        :param num_games: Number of games to play, defaults to 50
        :type num_games: int, optional
        """
        win_history = np.empty([0,])
        game = s.Surround(human_controls=0)
        game.reset()
        
        for _ in range(num_games):
            while True:
                model_action = self.play(game.board)
                adversary_action = adversary.play(game.board)
                _, _, _, lose1, lose2 = game.step((model_action, adversary_action))
                
                if lose1 and lose2:
                    win_history = np.append(win_history, 0)
                    break
                elif lose1:
                    win_history = np.append(win_history, -1)
                    break
                elif lose2:
                    win_history = np.append(win_history, 1)
                    break

        return win_history.mean()      
    
    def train(self, adversary: 'NeuralNet', min_win_rate: int = 0.6) -> None: # ATTENTION: docstring must be updated
        """Train the model until it reaches a win rate of 0.55 against the adversary.

        :param boards_buffer: _description_
        :type boards_buffer: np.array
        :param probs_buffer: _description_
        :type probs_buffer: np.array
        :param adversary: _description_
        :type adversary: NeuralNet
        :return: Win rate history
        :rtype: np.array
        """        
        win_rate = self.get_win_rate(adversary)

        if win_rate > min_win_rate:
            print(f'Win rate: {win_rate}')
            print("Win rate already satisfactory, skipping trainment...")

            return np.array([win_rate])
        else:
            win_rate_history = np.empty([win_rate])
        
            print(f'Win rate: {win_rate}')
            print("Trainment needed, proceding to trainment...")

            trainment_step = 1

            while win_rate < min_win_rate:
                boards_buffer, probs_buffer = MCTS.get_buffers(adversary) # ATTENTION: MCTS is not implemented yet

                self.fit(boards_buffer, probs_buffer, epochs=10)
                
                win_rate = self.get_win_rate(adversary)
                win_rate_history = np.append(win_rate_history, win_rate)

                print(f'Win rate: {win_rate}')
                print("Trainment step:", trainment_step)
                trainment_step += 1
            
            return win_rate_history
        
    def save(self, file_name: str) -> None:
        """Save the model.

        :param file_name: Name of the file to save the model
        :type file_name: str
        """            
        self.model.save(f'./saved_models/{file_name}.keras')

    def load(self, file_name: str) -> None:
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
