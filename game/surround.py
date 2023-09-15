import numpy as np
import pygame as pg
import time

BOARD_HEIGHT = 32
BOARD_WIDTH = 32

class HumanBoard():
    
    colors = [[84, 92, 214], [212, 108, 195], [200, 72, 72], [183, 194, 95]]
    height_score = 30
    height = BOARD_HEIGHT * 10 + height_score
    width = BOARD_WIDTH * 10

    def __init__(self, frame_rate: int) -> None:
        self.frame_rate = frame_rate

        pg.init()
        pg.display.set_caption("Surround")
        self.screen = pg.display.set_mode((self.width, self.height))
        self.font = pg.font.Font(None, 30)

    def render(self, board: np.array, lose1: bool, lose2: bool, score: tuple) -> None:
        self.screen.fill((0, 0, 0))
        score_text = self.font.render(f"Score: {score[0]} x {score[1]}",
                                       True, (255, 255, 255))
        self.screen.blit(score_text, (0, 0))

        for i in range(BOARD_WIDTH):
            for j in range(BOARD_HEIGHT):
                pg.draw.rect(self.screen,
                              self.colors[board[i][j]],
                              (i * 10, j * 10 + self.height_score, 10, 10))
        
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
        win_text = self.font.render(f"Player {player} win!",
                                       True, (255, 255, 255))
        self.screen.blit(win_text, (self.width//2, self.height//2))
        pg.display.update()
        time.sleep(1)

    def close(self) -> None:
        pg.quit()
    
class Player():
    def __init__(self, pos_x: int, pos_y: int, init_action: int, value: int) -> None:
        f""" Init a player

        :param pos_x: Init x position, between 0 and {BOARD_WIDTH - 1}
        :type pos_x: int
        :param pos_y: Init y position, between 0 and {BOARD_HEIGHT - 1}
        :type pos_y: int
        :param init_action: Init action, 1 = right, 2 = up, 3 = left, 4 = down
        :type init_action: int
        :param value: Value of the player
        :type value: int
        """        
        self.value = value
        self.pos_x = pos_x
        self.pos_y = pos_y
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
            board[self.pos_x, self.pos_y] = 1

            self.pos_x += x_atualization
            self.pos_y += y_atualization

            board[self.pos_x, self.pos_y] = self.value
        
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

class Surround():
    f"""Surround game environment
    
    Attributes:
        action_space: list of possible actions (0 = stay, 1 = right, 2 = up, 3 = left, 4 = down)
        score: tuple of player1 and player2 score
        board: numpy array ({BOARD_WIDTH}, {BOARD_HEIGHT}) of the board
        player1: player1 object
        player2: player2 object
    Methods:
        reset: reset the game environment
        step: make a move in the game environment
    """
    action_space = [0, 1, 2, 3, 4]
    score = (0, 0)

    def __init__(self, human_render: bool = False, frame_rate: int = 1) -> None:
        """Init the game environment and game mode

        :param human_render: Needs human visualization, defaults to False
        :type human_render: bool, optional
        :param frame_rate: Frame rate for human visualization, defaults to 1
        :type frame_rate: int, optional
        """        
        self.human_render = human_render
        self.frame_rate = frame_rate

        if self.human_render:
            self.human_board = HumanBoard(self.frame_rate)

    def reset(self) -> None:
        """Reset the game enviroment (and initialize)"""
        # 0 = empty
        self.board = np.zeros((BOARD_WIDTH, BOARD_HEIGHT), dtype = int)
        # 1 = wall
        self.board[0, :] = 1
        self.board[BOARD_WIDTH - 1, :] = 1
        self.board[:, 0] = 1
        self.board[:, BOARD_HEIGHT - 1] = 1
        # 2 = player1
        self.player1 = Player(BOARD_WIDTH * 1//4, BOARD_HEIGHT//2, 1, 2)
        # 3 = player2
        self.player2 = Player(BOARD_WIDTH * 3//4, BOARD_HEIGHT//2, 3, 3)

    def check_lose(self, old_board) -> tuple:
        """Check if one of the players lose

        :param old_board: Board before the move
        :type old_board: np.array
        :return: Tuple of player1 lose and player2 lose
        :rtype: tuple(bool, bool)
        """
        lose1 = False
        lose2 = False

        if old_board[self.player1.pos_x, self.player1.pos_y] != 0:
            lose1 = True
        if old_board[self.player2.pos_x, self.player2.pos_y] != 0:
            lose2 = True
        return lose1, lose2

    def step(self, action: tuple) -> tuple:
        """Make a move in the game environment

        :param action: Tuple of player1 and player2 actions on the action space
        :type action: tuple(int, int)
        :return: Tuple of the board, player1 lose and player2 lose
        :rtype: tuple(np.array, bool, bool)
        """
        # Make the players movement
        if action[0] not in self.action_space or action[1] not in self.action_space:
            raise OutOfActionSpace
        
        old_board = self.board.copy()

        self.board = self.player1.move(self.board, action[0])
        self.board = self.player2.move(self.board, action[1])

        lose1, lose2 = self.check_lose(old_board)

        # Render the game if human visualization is needed
        if self.human_render:
            self.human_board.render(self.board, lose1, lose2, self.score)

        # Check if the game is over
        if lose1 and lose2:
            # Tie
            self.human_board.win(0)
            self.reset()
        elif lose1:
            # Player2 win
            self.human_board.win(2)
            self.score = (self.score[0], self.score[1] + 1)
            self.reset()
        elif lose2:
            # Player1 win
            self.human_board.win(1)
            self.score = (self.score[0] + 1, self.score[1])
            self.reset()
        print(lose1, lose2)

        return self.board, lose1, lose2

if __name__ == "__main__":
    jogo = Surround(human_render=True, frame_rate=10)
    jogo.reset()
    x = 1
    while x < 100:
        y = 1
        while y < 100:
            acao = (0,1)
            jogo.step(acao)
            y += 1
        x += 1
        print(x)