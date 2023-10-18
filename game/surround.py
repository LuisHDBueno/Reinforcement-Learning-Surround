import numpy as np
import pygame as pg
import time
import math
import os
import sys
import tensorflow as tf

# sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

# import net_models as nm

BOARD_HEIGHT = 16
BOARD_WIDTH = 16
FONT = 30

class HumanBoard():
    """Class to render the game in a pygame window

    Attributes:
        colors: list of colors for each element in the board,
            (0 = empty, 1 = wall, 2 = player1, 3 = player2)
        height_score: height of the score text
        height: height of the pygame window
        width: width of the pygame window
        screen: pygame screen
        font: pygame font
    Methods:
        render: render the pygame window
        win: render the win text
        close: close the pygame window
    """
    pixel_size = 20    
    colors = [[84, 92, 214], [212, 108, 195], [200, 72, 72], [183, 194, 95]]
    height_score = 30
    height = BOARD_HEIGHT * pixel_size + height_score
    width = BOARD_WIDTH * pixel_size

    def __init__(self, frame_rate: int) -> None:
        """Init the pygame window

        :param frame_rate: Frame rate for human visualization
        :type frame_rate: int
        """        
        self.frame_rate = frame_rate

        pg.init()
        pg.display.set_caption("Surround")
        self.screen = pg.display.set_mode((self.width, self.height))
        self.font = pg.font.Font(None, FONT)

    def render(self, board: np.array, score: tuple) -> None:
        """ Render the window

        :param board: Board of the game
        :type board: np.array
        :param score: Tuple of player1 and player2 score
        :type score: tuple
        """        
        self.screen.fill((0, 0, 0))
        score_text = self.font.render(f"Score: {score[0]} x {score[1]}",
                                       True, (255, 255, 255))
        self.screen.blit(score_text, (0, 0))

        for i in range(BOARD_WIDTH):
            for j in range(BOARD_HEIGHT):
                pg.draw.rect(self.screen,
                              self.colors[0],
                              (i * self.pixel_size,
                                j * self.pixel_size + self.height_score,
                                self.pixel_size, self.pixel_size))
                if board[i, j, 0] == 1:
                    pg.draw.rect(self.screen,
                                  self.colors[1],
                                  (i * self.pixel_size,
                                    j * self.pixel_size + self.height_score,
                                    self.pixel_size, self.pixel_size))
                if board[i, j, 1] == 1:
                    pg.draw.rect(self.screen,
                                  self.colors[2],
                                  (i * self.pixel_size,
                                    j * self.pixel_size + self.height_score,
                                    self.pixel_size, self.pixel_size))
                if board[i, j, 2] == 1:
                    pg.draw.rect(self.screen,
                                  self.colors[3],
                                  (i * self.pixel_size,
                                    j * self.pixel_size + self.height_score,
                                    self.pixel_size, self.pixel_size))
        
        pg.display.update()
        time.sleep(1/self.frame_rate)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

    def win(self, player: int) -> None:
        """ Render the win text

        :param player: _description_
        :type player: int
        """
        if player == 0:
            win_text = self.font.render("Tie!", True, (255, 255, 255))
            self.screen.blit(win_text, (self.width//2, self.height//2))
        else:        
            win_text = self.font.render(f"Player {player} win!",
                                        True, (255, 255, 255))
            self.screen.blit(win_text, (self.width//3, self.height//2))
        pg.display.update()
        time.sleep(1)

    def close(self) -> None:
        """Close the pygame window"""        
        pg.quit()
    
class Player():
    def __init__(self, pos_x: int, pos_y: int, layer: int, init_action: int) -> None:
        f""" Init a player

        :param pos_x: Init x position, between 0 and {BOARD_WIDTH - 1}
        :type pos_x: int
        :param pos_y: Init y position, between 0 and {BOARD_HEIGHT - 1}
        :type pos_y: int
        :param layer: Layer of the player in the board, 1 = player1, 2 = player2
        :param init_action: Init action, 1 = right, 2 = up, 3 = left, 4 = down
        :type init_action: int
        """        
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.layer = layer
        self.last_action = init_action

    def move(self, board: np.array, action: int) -> tuple:
        """ Make a move in the board

        :param board: Board of the game
        :type board: np.array
        :param action: Action to be made, 0 = stay, 1 = right, 2 = up, 3 = left, 4 = down
        :type action: int
        :return: Tuple of the new board and if the player lose
        :rtype: tuple
        """
        # Auxiliar
        
        def atualize_pos(x_atualization: int, y_atualization: int) -> None:
            """Atualize the player position

            :param x_atualization: number of steps in x axis
            :type x_atualization: int
            :param y_atualization: number of steps in y axis
            :type y_atualization: int
            """

            # Update the player position
            board[self.pos_x, self.pos_y, self.layer] = 0

            self.pos_x += x_atualization
            self.pos_y += y_atualization
            if self.pos_x > BOARD_WIDTH - 1:
                self.pos_x = BOARD_WIDTH - 1
            if self.pos_y > BOARD_HEIGHT - 1:
                self.pos_y = BOARD_HEIGHT - 1
            # Fild the wall layer            
            board[self.pos_x, self.pos_y, 0] = 1
            board[self.pos_x, self.pos_y, self.layer] = 1
        
        match action:
            case 0:
                # If the player stay, continue in the same direction
                self.move(board, self.last_action)
            case 1:
                # Check if the player try to go back
                if self.last_action == 3:
                    # If yes, continue in the same direction
                    board= self.move(board, 3)
                else:
                    atualize_pos(1, 0)
                    self.last_action = 1

            case 2:
                if self.last_action == 4:
                    board = self.move(board, 4)
                else:
                    atualize_pos(0, 1)
                    self.last_action = 2

            case 3:
                if self.last_action == 1:
                    board = self.move(board, 1)
                else:
                    atualize_pos(-1, 0)
                    self.last_action = 3

            case 4:
                if self.last_action == 2:
                    board = self.move(board, 2)
                else:
                    atualize_pos(0, -1)
                    self.last_action = 4
        
        return board

class OutOfActionSpace(Exception):
    def __init__(self) -> None:
        super().__init__("Action not in action space") 

class HumanControls():
    """Class to control the human players
    Only works if the game was reseted

    Attributes:
        player: player to be controlled
    Methods:
        get_action: get the action of the player
    """    
    def __init__(self, n_player: int) -> None:
        """Init the player to be controlled

        :param n_player: number of the player to be controlled
        :type n_player: int
        """
        if n_player == 1:
            self.control = [self.get_action1]
        elif n_player == 2:
            self.control = [self.get_action1, self.get_action2]
        else:
            raise Exception("Invalid player number")
        
    def get_action1(self) -> int:
        """Get the action of the player 1

        :return: Action of the player
        :rtype: int
        """
        event = pg.key.get_pressed()
        if event[pg.K_d]:
            return 1
        elif event[pg.K_s]:
            return 2
        elif event[pg.K_a]:
            return 3
        elif event[pg.K_w]:
            return 4
        return 0
    
    def get_action2(self) -> int:
        """Get the action of the player 2

        :return: Action of the player
        :rtype: int
        """
        event = pg.key.get_pressed()
        if event[pg.K_RIGHT]:
            return 1
        elif event[pg.K_DOWN]:
            return 2
        elif event[pg.K_LEFT]:
            return 3
        elif event[pg.K_UP]:
            return 4
        return 0
    
    def get_moves(self):
        move = []
        for control in self.control:
            move.append(control())
        return move

class Surround():
    f"""Surround game environment
    
    Attributes:
        action_space: list of possible actions (0 = stay, 1 = right, 2 = up, 3 = left, 4 = down)
        score: tuple of player1 and player2 score
        board: numpy array ({BOARD_WIDTH}, {BOARD_HEIGHT}, 3) of the board.
        First dimension is for the x axis, second for the y axis and third for
        the elements wall, player1 and player2 (in this order)
        player1: player1 object
        player2: player2 object
        human_render: Needs human visualization
        human_controls: Number of human players
        frame_rate: Frame rate for human visualization
        lose1: If player1 lose
        lose2: If player2 lose
    Methods:
        reset: reset the game environment
        step: make a move in the game environment
    """
    action_space = [0, 1, 2, 3, 4]
    score = (0, 0)

    def __init__(self, human_render: bool = False,
                 human_controls: int = 0, frame_rate: int = 1) -> None:
        """Init the game environment and game mode

        :param human_render: Needs human visualization, defaults to False
        :type human_render: bool, optional
        :param human_controls: Number of human players, defaults to 0
        :type human_controls: int, optional
        :param frame_rate: Frame rate for human visualization, defaults to 1
        :type frame_rate: int, optional
        """        
        self.human_render = human_render
        self.frame_rate = frame_rate
        self.lose1 = False
        self.lose2 = False
        self.reward = 0

        if self.human_render:
            self.human_board = HumanBoard(self.frame_rate)
        if human_controls != 0:
            self.human_controls = HumanControls(human_controls)
        else:
            self.human_controls = None

    def reset(self) -> None:
        """Reset the game enviroment (and initialize)"""
        # 0 = empty
        self.board = np.zeros((BOARD_WIDTH, BOARD_HEIGHT, 3), dtype = np.bool_)
        # First layer is the walls
        self.board[0, : , 0] = 1
        self.board[BOARD_WIDTH - 1, : , 0] = 1
        self.board[ : , 0, 0] = 1
        self.board[ : , BOARD_HEIGHT - 1, 0] = 1
        # Second layer is the player 1
        self.player1 = Player(BOARD_WIDTH * 1//4, BOARD_HEIGHT//2, 1, 1)
        # Third layer is the player 2
        self.player2 = Player(BOARD_WIDTH * 3//4, BOARD_HEIGHT//2, 2, 3)
        self.board[self.player1.pos_x, self.player1.pos_y, 1] = 1
        self.board[self.player2.pos_x, self.player2.pos_y, 2] = 1

    def check_lose(self, old_board) -> tuple:
        """Check if one of the players lose

        :param old_board: Board before the move
        :type old_board: np.array
        :return: Tuple of player1 lose and player2 lose
        :rtype: tuple(bool, bool)
        """
        lose1 = False
        lose2 = False

        if old_board[self.player1.pos_x, self.player1.pos_y, 0] != 0:
            lose1 = True
        if old_board[self.player2.pos_x, self.player2.pos_y, 0] != 0:
            lose2 = True
        return lose1, lose2

    def copy(self):
        return (self.board.copy(), (self.player1.pos_x, self.player1.pos_y),
                (self.player2.pos_x, self.player2.pos_y), (self.lose1, self.lose2), self.reward, self.score)
    
    def set_state(self, state):

        self.board = state[0]
        self.player1.pos_x, self.player1.pos_y = state[1]
        self.player2.pos_x, self.player2.pos_y = state[2]
        self.lose1, self.lose2 = state[3]
        self.reward = state[4]
        self.score = state[5]
    def step(self, action: tuple) -> tuple:
        """Make a move in the game environment

        :param action: Tuple of player1 and player2 actions on the action space.
        If number of human players is 1, the second action is ignored.
        If number of human players is 2, both actions are ignored.
        :type action: tuple(int, int)
        :return: Tuple of the board, player1 lose and player2 lose
        :rtype: tuple(np.array, bool, bool)
        """
        # Make the players movement
        if action[0] not in self.action_space or action[1] not in self.action_space:
            raise OutOfActionSpace
        
        old_board = self.board.copy()

        if self.human_controls != None:
            human_moves = self.human_controls.get_moves()
            if len(human_moves) == 1:
                action = (human_moves[0], action[1])
            else:
                action = human_moves
        
        self.board = self.player1.move(self.board, action[0])
        self.board = self.player2.move(self.board, action[1])

        lose1, lose2 = self.check_lose(old_board)

        # Render the game if human visualization is needed
        if self.human_render:
            self.human_board.render(self.board, self.score)

            # Check if the game is over
            if lose1 and lose2:
                # Tie
                self.human_board.win(0)
                self.reset()
                reward = -10
            elif lose1:
                # Player2 win
                self.human_board.win(2)
                self.score = (self.score[0], self.score[1] + 1)
                self.reset()
                reward = -100
            elif lose2:
                # Player1 win
                self.human_board.win(1)
                self.score = (self.score[0] + 1, self.score[1])
                self.reset()
                reward = 100
            else:
                reward = 0
        else:
            if lose1 and lose2:
                # Tie
                self.reset()
                reward = -10
            elif lose1:
                # Player2 win
                self.score = (self.score[0], self.score[1] + 1)
                self.reset()
                reward = -100
            elif lose2:
                # Player1 win
                self.score = (self.score[0] + 1, self.score[1])
                self.reset()
                reward = 100
            else:
                reward = 0
        self.lose1 = lose1
        self.lose2 = lose2
        self.reward = reward
        return reward, old_board, self.board, lose1, lose2, action

class Encoding:
    """Class to encode and decode the game state
    """
    def __init__(self) -> None:
        pass

    def encode(self, board: np.array) -> int:
        """Encode the board into a int

        :param board: Board of the game
        :type board: np.array
        :return: Int representing the board
        :rtype: int
        """
        # Convert matrices to binary strings
        game_state_bits = board[:,:,0][1:-1,1:-1].flatten().astype(np.uint8)
        player1 = np.nonzero(board[:,:,1])
        player2 = np.nonzero(board[:,:,2])
        player1x, player1y = player1[0][0], player1[1][0]
        player2x, player2y = player2[0][0], player2[1][0]
        pos_size = math.ceil(np.log2(BOARD_WIDTH))
        player1x_bits = np.binary_repr(player1x, width=pos_size)
        player1y_bits = np.binary_repr(player1y, width=pos_size)
        player2x_bits = np.binary_repr(player2x, width=pos_size)
        player2y_bits = np.binary_repr(player2y, width=pos_size)
        # Combine game state and player positions into a single binary string
        encoded_bits = np.zeros((len(game_state_bits) + pos_size * 4,), dtype=np.uint8)
        board_size = (BOARD_WIDTH - 2)*(BOARD_HEIGHT-2)
        encoded_bits[board_size-len(game_state_bits):board_size] = game_state_bits
        encoded_bits[board_size:board_size+pos_size] |= np.array(list(player1x_bits), dtype=np.uint8)
        encoded_bits[board_size+pos_size:board_size+2*pos_size] |= np.array(list(player1y_bits), dtype=np.uint8)
        encoded_bits[board_size+2*pos_size:board_size+3*pos_size]  |= np.array(list(player2x_bits), dtype=np.uint8)
        encoded_bits[board_size+3*pos_size:]  |= np.array(list(player2y_bits), dtype=np.uint8)
        encoded_int = encoded_int = int(''.join(map(str, encoded_bits.tolist())), 2)
        return encoded_int

    def decode(self, state_int: int) -> np.array:
        """Decode the int into a board

        :param state_int: Int representing the board
        :type state_int: int
        :return: Board of the game
        :rtype: np.array
        """
        binary_str = bin(state_int)[2:]
        pos_size = math.ceil(np.log2(BOARD_WIDTH))
        board_size = (BOARD_WIDTH - 2)*(BOARD_HEIGHT-2)
        binary_str = '0'*(board_size + pos_size*4 - len(binary_str)) + binary_str
        game_state_str = binary_str[:board_size]
        player1x = binary_str[board_size:
                            board_size+pos_size]
        player1y = binary_str[board_size+pos_size:
                            board_size+2*pos_size]
        player2x = binary_str[board_size+2*pos_size:
                            board_size+3*pos_size]
        player2y = binary_str[board_size+3*pos_size:]

        # Convert game state string to matrix with and added border of ones
        game_state = np.ones((BOARD_WIDTH, BOARD_HEIGHT), dtype=np.bool_)
        game_state[1:-1,1:-1] = np.array(list(map(int, game_state_str))).reshape(BOARD_WIDTH-2, BOARD_HEIGHT-2)

        # Convert player positions to matrix
        player1 = np.zeros((BOARD_WIDTH, BOARD_HEIGHT), dtype=np.bool_)
        player1[int(player1x, 2), int(player1y, 2)] = 1
        player2 = np.zeros((BOARD_WIDTH, BOARD_HEIGHT), dtype=np.bool_)
        player2[int(player2x, 2), int(player2y, 2)] = 1

        # Concatenate matrices
        board = np.stack((game_state, player1, player2), axis=2)

        return board

    
if __name__ == "__main__":
    jogo = Surround(human_render=True, human_controls=0, frame_rate=2)
    jogo.reset()

    model = nm.ConvolutionNet()
    model.load('cnn')
    for _ in range(10):
        while True:
            retorno = np.random.randint(5)
            a = model.play(jogo.board)
            reward, old_board, board, lose1, lose2 = jogo.step((a, retorno))
            print(a, reward, lose1, lose2)
            
            if lose1 or lose2:
                break