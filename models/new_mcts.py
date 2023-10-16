import numpy as np
from copy import deepcopy
import sys
import os
import random
import time
import net_models as nm
from queue import Queue
sys.path.append(os.path.join(os.path.dirname(__file__), '../game'))
import surround as s

class MCTS():

    def __init__(self, player:nm.NeuralNet, adversary:nm.NeuralNet) -> None:
        self.player = player
        self.adversary = adversary
        self.tree = {}
        self.game = s.Surround()
        self.game.reset()

    """{1: no, 2: no, 3: no, 4: no, acao: int, recompensa: int}"""

    def reset(self) -> None:
        self.tree = None
        self.game.reset()

    def terminal_game(self, game:s.Surround) -> int:

        while not (game.lose1 or game.lose2):
            reward, *_ = game.step(self.player.play(game.board), self.adversary.play(game.board))

        return reward
    
    def get_buffers(self):
        boards_buffer = []
        probs_buffer = []

        # Grow the tree
        
        boards_buffer = np.array(boards_buffer)
        probs_buffer = np.array(probs_buffer)
        return boards_buffer, probs_buffer