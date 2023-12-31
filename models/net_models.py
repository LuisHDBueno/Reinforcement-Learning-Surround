import numpy as np
import tensorflow as tf
from keras.models import Sequential
import mcts as m
from keras.layers import Conv2D, Flatten, Dense
import os
import sys
from copy import deepcopy

sys.path.append(os.path.join(os.path.dirname(__file__), '../game'))

import surround as s

BOARD_HEIGHT = s.BOARD_HEIGHT
BOARD_WIDTH = s.BOARD_WIDTH

# Permutation matrix to flip the board
PERMUTATION_MATRIX = np.zeros((BOARD_HEIGHT, BOARD_WIDTH))
for i in range(BOARD_HEIGHT):
    PERMUTATION_MATRIX[i, BOARD_WIDTH - 1 - i] = 1

class NeuralNet():
    """Generic neural network class. Implements commom methods for all neural networks.
    
    Methods:
        predict: Predict the reward distribution for each action.
        fit: Fit the model with the given boards and rewards.
        check_out_of_bounds: Check if the position is out of bounds.
        legal_moves: Returns a list of legal moves for both players.
        play: Choose the best action to play.
        get_win_rate: Get the win rate of the model against a fixed adversary.
        train: Train the model until it reaches a win rate of min_win_rate against the adversary.
        save: Save the model.
        load: Load pre-trained model.
    """
    def predict(self, board: np.array) -> np.array:
        """Predict the reward distribution for each action.

        :param board: Current game board
        :type board: np.array
        :return: Array containing the relative reward for each action, interpreted as a probability distribution
        :rtype: np.array
        """         
        return self.model.predict(board.reshape(1, BOARD_WIDTH, BOARD_HEIGHT, 3), verbose=0)
        
    def fit(self, boards: np.array, rewards: np.array, batch_size: int = 32, epochs: int = 1) -> None:
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
        if len(boards) == 0:
            print("No boards to fit")
            return
        self.model.fit(boards, rewards, batch_size=batch_size, epochs=epochs, verbose=1)

    def check_out_of_bounds(self, pos_x: int, pos_y: int) -> bool:
        """Check if the position is out of bounds.

        :param pos_x: x position
        :type pos_x: int
        :param pos_y: y position
        :type pos_y: int
        :return: True if the position is out of bounds, False otherwise
        :rtype: bool
        """
        if pos_x > BOARD_WIDTH - 1:
            return True
        if pos_y > BOARD_HEIGHT - 1:
            return True
        return False
    
    def legal_moves(self,game: s.Surround) -> list[tuple[int]]:
        """Returns a list of legal moves for both players

        :param game: game state
        :type game: s.Surround
        :return: list of legal moves for both players
        :rtype: list[tuple[int]]
        """        
        board = game.board[:,:,0]
        player1 = game.player1.pos_x, game.player1.pos_y
        player2 = game.player2.pos_x, game.player2.pos_y
        moves1 = []
        moves2 = []

        if not self.check_out_of_bounds(player1[0]+1, player1[1]):
            if not (board[player1[0]+1, player1[1]]): moves1.append(1)
        if not self.check_out_of_bounds(player1[0], player1[1]+1):
            if not (board[player1[0], player1[1]+1]): moves1.append(2)
        if not self.check_out_of_bounds(player1[0]-1, player1[1]):
            if not (board[player1[0]-1, player1[1]]): moves1.append(3)
        if not self.check_out_of_bounds(player1[0], player1[1]-1):
            if not (board[player1[0], player1[1]-1]): moves1.append(4)

        if not self.check_out_of_bounds(player2[0]+1, player2[1]):
            if not (board[player2[0]+1, player2[1]]): moves2.append(1)
        if not self.check_out_of_bounds(player2[0], player2[1]+1):
            if not (board[player2[0], player2[1]+1]): moves2.append(2)
        if not self.check_out_of_bounds(player2[0]-1, player2[1]):
            if not (board[player2[0]-1, player2[1]]): moves2.append(3)
        if not self.check_out_of_bounds(player2[0], player2[1]-1):
            if not (board[player2[0], player2[1]-1]): moves2.append(4)

        return moves1, moves2

    def play(self, game, player: int = 1) -> int:
        """Choose the best action to play.

        :param game: Game state
        :type game: s.Surround
        :param player: Player to play, defaults to 1
        :type player: int, optional
        :return: Best action to play
        :rtype: int
        """
        board = game.board
        moves1, moves2 = self.legal_moves(game)

        if player == 1:
            predict = self.predict(board).reshape(4,)

            # Add noise to the probabilities
            fuzzy = predict + 0.5 * np.random.dirichlet(np.array([0.03]*4)).reshape(4,)          
            best_action = np.argmax(fuzzy) + 1

            # Check if is legal
            if best_action in moves1:
                best_action = best_action
            elif moves1 == []:
                best_action = 0
            else:
                best_action = np.random.choice(moves1)

        elif player == 2:
            # Flip board
            board = deepcopy(board)
            board[:,:,0] = np.matmul(board[:,:,0], PERMUTATION_MATRIX)
            board[:,:,1] = np.matmul(board[:,:,1], PERMUTATION_MATRIX)
            board[:,:,2] = np.matmul(board[:,:,2], PERMUTATION_MATRIX)

            # Swap players' layers
            board[:,:,1], board[:,:,2] = board[:,:,2], board[:,:,1]
            
            predict = self.predict(board).reshape(4,)

            fuzzy = predict + 0.5 * np.random.dirichlet(np.array([0.03]*4)).reshape(4,)          
            best_action = np.argmax(fuzzy) + 1

            # Free flipped board from memory
            del board
            
            if best_action in moves2:
                best_action = best_action
            elif moves2 == []:
                best_action = game.player2.last_action
            else:
                best_action = np.random.choice(moves2)

        return best_action
    
    def get_win_rate(self, adversary: 'NeuralNet', num_games: int = 100) -> float:
        """Get the win rate of the model against a fixed adversary.

        :param adversary: Neural network to play against
        :type adversary: NeuralNet
        :return: Win rate of the model against the adversary
        :rtype: float
        :param mcts: Monte Carlo Tree Search object, defaults to None
        :type mcts: m.MCTS, optional
        :param num_games: Number of games to play, defaults to 50
        :type num_games: int, optional
        :param search_count: Number of searches to perform in each game, defaults to 1000
        :type search_count: int, optional
        """
        win_history = []
        game = s.Surround(human_controls=0)
        game.reset()

        # Play num_games games to get the win rate
        for _ in range(num_games):
            while True:
                model_action = self.play(game)
                adversary_action = adversary.play(game, 2)
                _, _, _, lose1, lose2, _ = game.step((model_action, adversary_action))

                if lose1 and lose2:
                    win_history.append(0)
                    break
                elif lose1:
                    win_history.append(-1)
                    break
                elif lose2:
                    win_history.append(1)
                    break

        mean = np.mean(win_history)
        return mean
    
    def train(self, adversary: 'NeuralNet', min_win_rate: int = 0.2) -> np.array:
        """Train the model until it reaches a win rate of min_win_rate against the adversary.

        :param adversary: Neural network to play against
        :type adversary: NeuralNet
        :param min_win_rate: Minimum win_rate to stop trainment, defaults to 0.2
        :type min_win_rate: int, optional
        :return: Win rate history of the trainment
        :rtype: np.array
        """        
        win_rate = self.get_win_rate(adversary)

        if win_rate > min_win_rate:
            print(f'Win rate: {win_rate}')
            print("Win rate already satisfactory, skipping trainment...")

            return np.array([win_rate])
        else:
            mcts = m.MCTS(self, adversary)
            win_rate_history = np.array([win_rate])
            print(f'Win rate: {win_rate}')
            print("Trainment needed, proceding to trainment...")

            trainment_step = 1
            boards_buffer = []
            probs_buffer = []

            while win_rate < min_win_rate:
                print("====\nTrainment step:", trainment_step)

                # Get trainment data
                boards_buffer, probs_buffer = mcts.get_buffers(boards_buffer=boards_buffer, probs_buffer=probs_buffer)
                array_boards_buffer = np.array(boards_buffer)
                array_probs_buffer = np.array(probs_buffer)

                # One hot encode the probabilities
                one_hot_probs = np.zeros((array_probs_buffer.shape[0], 4))
                one_hot_probs[np.arange(array_probs_buffer.shape[0]), np.argmax(array_probs_buffer, axis=1)] = 1
                array_probs_buffer = one_hot_probs

                self.fit(array_boards_buffer, array_probs_buffer, epochs=10)
                
                win_rate = self.get_win_rate(adversary)
                win_rate_history = np.append(win_rate_history, win_rate)

                print(f'Win rate: {win_rate}')
                trainment_step += 1

            # Free memory
            del mcts
            del boards_buffer
            del probs_buffer

            return win_rate_history
        
    def save(self, file_name: str) -> None:
        """Save the model.

        :param file_name: Name of the file to save the model
        :type file_name: str
        """            
        self.model.save(f'{file_name}.keras')

    def load(self, file_name: str) -> None:
        """Load pre-trained model.

        :param file_name: Name of the file to load the model
        :type file_name: str
        """            
        self.model = tf.keras.models.load_model(f'./saved_models/{file_name}.keras')

class DenseNet(NeuralNet):
    """Dense neural network class. Implements a dense neural network with n_layers hidden layers and n_neurons neurons per layer.
    
    Attributes:
        model: Keras model.
    """
    def __init__(self,  n_layers: int = 5, n_neurons: int = 256,
                input_shape: tuple = (BOARD_WIDTH, BOARD_HEIGHT, 3)) -> None:
        """Generate a dense neural network with n_layers hidden layers and 256 neurons per layer.

        :param n_layers: Number of Hidden layers, defaults to 5
        :type n_layers: int, optional
        :param n_neurons: Number of neurons by layer, defaults to 256
        :type n_neurons: int, optional
        :param input_shape: Shape of the board, defaults to (BOARD_WIDTH, BOARD_HEIGHT, 3)
        :type input_shape: tuple, optional
        """                   
        self.model = Sequential()
        # Input layer
        self.model.add(Flatten(input_shape=input_shape))
        # Hidden layers
        for _ in range(n_layers - 1):
            self.model.add(Dense(n_neurons, activation='relu',
                                  kernel_regularizer = tf.keras.regularizers.l2(0.0001)))
        # Output layer
        self.model.add(Dense(4, activation='softmax'))
        self.model.compile(optimizer='adam', loss="categorical_crossentropy",
                            metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.CategoricalCrossentropy()], run_eagerly=True)

class ConvolutionNet(NeuralNet):
    """Convolutional neural network class. Implements a convolutional neural network with n_layers hidden layers and n_neurons neurons per layer.
    
    Attributes:
        model: Keras model.
    """
    def __init__(self, n_conv_layers: int = 6, n_dense_layers: int = 3, n_neurons: int = 256,
                input_shape: tuple = (BOARD_WIDTH, BOARD_HEIGHT, 3)) -> None:
        """Generate a convolutional neural network with n_layers hidden layers and 256 neurons per layer.

        :param n_conv_layers: Number of convolutional layers, defaults to 5
        :type n_conv_layers: int, optional
        :param n_dense_layers: Number of dense layers, defaults to 3
        :type n_dense_layers: int, optional
        :param n_neurons: Number of neurons by Dense layer, defaults to 256
        :type n_neurons: int, optional
        :param input_shape: Shape of the board, defaults to (BOARD_WIDTH, BOARD_HEIGHT, 3)
        :type input_shape: tuple, optional
        """

        self.model = Sequential()

        self.model.add(Conv2D(32, (4, 4), activation='relu', padding='same',
                                   input_shape = input_shape,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        
        for _ in range(n_conv_layers - 1):
            self.model.add(Conv2D(32, (4, 4), activation='relu', padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        
        self.model.add(Flatten())

        # Dense layers
        for _ in range(n_dense_layers - 1):
            self.model.add(Dense(n_neurons, activation='relu',
                                  kernel_regularizer = tf.keras.regularizers.l2(0.001)))
        
        # Output layer
        self.model.add(Dense(4, activation='softmax'))
        self.model.compile(optimizer='adam', loss="categorical_crossentropy",
                            metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.CategoricalCrossentropy()], run_eagerly=True)
