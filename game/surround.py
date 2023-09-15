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
        win_text = self.font.render(f"Player {player} win!",
                                       True, (255, 255, 255))
        self.screen.blit(win_text, (self.width//2, self.height//2))
        pg.display.update()
        time.sleep(1)

    def close(self) -> None:
        pg.quit()
    
class Player():
    def __init__(self, pos_x: int, pos_y: int, init_action: int, value: int) -> None:
        self.value = value
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.last_action = init_action

    def move(self, board, action: int) -> tuple:

        def check_lose(x_atualization: int, y_atualization: int) -> bool:
            if board[self.pos_x + x_atualization, self.pos_y + y_atualization] == 1:
                return True
            else:
                return False
        
        def atualize_pos(x_atualization: int, y_atualization: int) -> None:
            board[self.pos_x, self.pos_y] = 1

            self.pos_x += x_atualization
            self.pos_y += y_atualization

            board[self.pos_x, self.pos_y] = self.value

        match action:
            case 0:
                lose = False
                self.move(board, self.last_action)
            case 1:
                lose = check_lose(1, 0)
                if not lose:
                    atualize_pos(1, 0)
                    self.last_action = 1

            case 2:
                lose = check_lose(0, 1)
                if not lose:
                    atualize_pos(0, 1)
                    self.last_action = 2

            case 3:
                lose = check_lose(-1, 0)
                if not lose:
                    atualize_pos(-1, 0)
                    self.last_action = 3

            case 4:
                lose = check_lose(0, -1)
                if not lose:
                    atualize_pos(0, -1)
                    self.last_action = 4
        
        return board, lose

    
class Surround():
    action_space = [0, 1, 2, 3, 4]
    score = (0, 0)

    def __init__(self, human_render: bool = False, frame_rate: int = 1) -> None:
        self.human_render = human_render
        self.frame_rate = frame_rate
        if self.human_render:
            self.human_board = HumanBoard(self.frame_rate)

    def reset(self) -> None:
        self.board = np.zeros((BOARD_WIDTH, BOARD_HEIGHT), dtype = int)
        self.board[0, :] = 1
        self.board[BOARD_WIDTH - 1, :] = 1
        self.board[:, 0] = 1
        self.board[:, BOARD_HEIGHT - 1] = 1
        self.player1 = Player(BOARD_WIDTH * 1//4, BOARD_HEIGHT//2, 1, 2)
        self.player2 = Player(BOARD_WIDTH * 3//4, BOARD_HEIGHT//2, 3, 3)

    def step(self, action: tuple) -> tuple:
        self.board, lose1 = self.player1.move(self.board, action[0])
        self.board, lose2 = self.player2.move(self.board, action[1])

        if self.human_render:
            self.human_board.render(self.board, lose1, lose2, self.score)

        if lose1 and lose2:
            self.reset()
        elif lose1:
            self.score = (self.score[0], self.score[1] + 1)
            self.reset()
        elif lose2:
            self.score = (self.score[0] + 1, self.score[1])
            self.reset()

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