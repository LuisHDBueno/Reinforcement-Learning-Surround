import numpy as np
from copy import deepcopy
import sys
import os
import random
import time

from queue import Queue
sys.path.append(os.path.join(os.path.dirname(__file__), '../game'))
import surround as s

class MCTS():

    def __init__(self, player:'NeuralNet', adversary:'NeuralNet') -> None:
        """ Init the Monte Carlo Tree Search

        :param player: Neural Network to train
        :type player: NeuralNet
        :param adversary: Neural Network to play against
        :type adversary: NeuralNet
        """
        self.player = player
        self.adversary = adversary
        self.tree = {}
        # Game to be played
        self.game = s.Surround()
        self.game.reset()
        #Encoder of the board        
        self.encoder = s.Encoding()
        # Boards already in the tree
        self.boards_in_tree = set()

    """{1: no, 2: no, 3: no, 4: no, acao: int, recompensa: int}"""

    def reset(self) -> None:
        """Reset the game and the tree"""
        self.tree = {}
        self.game.reset()

    def terminal_game(self, game:s.Surround) -> int:
        """Play a game until it ends"""
        while ((not game.lose1) and (not game.lose2)):
            reward, *_ = game.step((self.player.play(game), self.adversary.play(game, player=2)))
        
        del game
        return reward
    
    def forbiden_move(self) -> int:
        """The forbiden move is the opposite of the last move of the player

        :return: Forbiden move in the action space
        :rtype: int
        """        
        last_action = self.game.player1.last_action
        if last_action == 1:
            return 3
        elif last_action == 2:
            return 4
        elif last_action == 3:
            return 1
        else:
            return 2
    
    def verify_chosen_action(self, game, action) -> bool:
        """ Verifies if the chosen action is already in the tree

        :param game: _description_
        :type game: _type_
        :param action: _description_
        :type action: _type_
        :return: _description_
        :rtype: bool
        """        
        board = game.board  
        ver_board = self.game.player1.move(board, action)
        ver_board = self.game.player2.move(ver_board, self.adversary.play(self.game, player=2))
        ver_board = self.encoder.encode(ver_board)
        if ver_board in self.boards_in_tree:
            return True
        else:
            return False
        
    def grow_tree(self) -> None:
        node = {}
        action_space = [1, 2, 3, 4]
        # Grow the tree
        self.boards_in_tree.add(self.encoder.encode(self.game.board))
        node["board"] = self.game.board
        forbiden_move = self.forbiden_move()
        node["forbiden_move"] = forbiden_move
        action_space.remove(forbiden_move)
        probs_action = self.player.predict(self.game.board)
        fuzzy_prob_actions = probs_action + np.random.dirichlet(np.array([0.03]*4)).reshape(4,) 
        chosen_action = np.argmax(probs_action) + 1

        ver_game = deepcopy(self.game)
        while self.verify_chosen_action(self.game, chosen_action):
            fuzzy_prob_actions += np.random.dirichlet(np.array([0.03]*4)).reshape(4,)
            chosen_action = np.argmax(probs_action) + 1
        del ver_game
        node["chosen_action"] = chosen_action

        for action in action_space:
            if action != chosen_action:
                if not (self.game.lose1 or self.game.lose2):
                    reward = self.terminal_game(deepcopy(self.game))
                else:
                    reward = self.game.reward
                node[action] = {"reward": reward}
        reward, *_ = self.game.step((action, self.adversary.play(self.game, player=2)))

        if (not (self.game.lose1 or self.game.lose2)):
            node[chosen_action] = self.grow_tree()
            node[forbiden_move] = self.game.player1.last_action
            return node
        else:
            node[chosen_action] = {"reward": reward}
            return node
        
    
    def get_node_reward(self, node):
        action_space = [1, 2, 3, 4]
        if "reward" in node.keys():
            return node["reward"]
        else:
            mean_reward = 0
            action_space.remove(node["forbiden_move"])
            for action in action_space:
                mean_reward += self.get_node_reward(node[action])
            mean_reward = mean_reward / len(action_space)
            node["reward"] = mean_reward
            return node["reward"]
    
    def tree_to_list(self) -> list:
        boards_buffer = []
        probs_buffer = []
        node = self.tree
        
        while type(node) == type({1: 2}):
            # print("\n\n O NÓ:")
            # print(node)
            # print(type(node))
            # print(node["board"] if "board" in node.keys() else "None")
            # print(list(node.keys()))
            # print("\n\n")

            if len(node.keys()) == 1:
                break
            boards_buffer.append(node["board"])
            node_probs = np.zeros((4,), dtype=np.float32)
            action_space = [1, 2, 3, 4]
            action_space.remove(node["forbiden_move"])
            for action in action_space:
                node_probs[action - 1] = self.get_node_reward(node[action])
            forbiden = node["forbiden_move"]
            if forbiden == 1:
                node_probs[node["forbiden_move"] - 1] = node_probs[2]
            elif forbiden == 2:
                node_probs[node["forbiden_move"] - 1] = node_probs[3]
            elif forbiden == 3:
                node_probs[node["forbiden_move"] - 1] = node_probs[0]
            else:
                node_probs[node["forbiden_move"] - 1] = node_probs[1]
                
            probs_buffer.append(node_probs)
            chosen_action = node["chosen_action"]
            node = node[chosen_action]
        return boards_buffer, probs_buffer


    def get_buffers(self, boards_buffer:list, probs_buffer:list):

        size = len(boards_buffer)
        while len(boards_buffer) < size + 50:
            # print("tamanho do buffer: ", len(boards_buffer))
            self.reset()
            self.tree = self.grow_tree()
            b_buffer, p_buffer = self.tree_to_list()
            boards_buffer.extend(b_buffer)
            probs_buffer.extend(p_buffer)

        return boards_buffer, probs_buffer