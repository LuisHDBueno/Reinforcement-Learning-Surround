
import numpy as np
from collections import Iterable
from copy import deepcopy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../game'))
import surround as s

class Node:
    """Node of MCTS
    :param move: moves taken from parent(move_player1, move_player2)
    :type move: tuple
    :param parent: parent node
    :type parent: Node
    :param N: number of simulations in this state
    :type N: int
    :param Q: Amount of wins in this state for each player
    :type Q: np.array[float]
    :param children: children nodes
    :type children: dict[move,node]
    :param outcome: -1 if not terminal, 0 if draw, 1 if p1 wins, 2 if p2 wins
    :type outcome: int
    """    
    def __init__(self, move:tuple(int),parent: Node):
        """init of class Node

        :param move: moves taken from parent(move_player1, move_player2)
        :type move: tuple
        :param parent: parent node
        :type parent: Node
        """        
        self.move = move
        self.parent = parent
        self.N = 0 # number of simulations
        self.Q = np.zeros((2,), dtype = np.int) # number of wins for each player
        self.children = {}#key: move, item: child Node
        self.outcome = -1 #-1 if not terminal, 0 if draw, 1 if p1 wins, 2 if p2 wins

    def add_children(self, children: Iterable) -> None:
        """add children. 
        There should be children for all legal moves

        :param children: children nodes to be Added
        :type children: Iterable
        """        
        for child in children:
            self.children[child.move] = child
        
    def ucb1(self, player:bool) -> float:
        """calculate index based policy USB1-Tuned for given player 
        :param totalN: total number of
        :param player: 0 for player 1, 1 for player 2
        :type player: bool
        :return: policy value
        :rtype: float
        """        
        return self.Q/self.N + np.sqrt(2*(np.log(self.parent.N))/self.N)

    def value(self) -> tuple[float]:
        """ calculates selection value

        :return: selection value of each player
        :rtype: tuple(float,float)
        """        

        return (self.ucb1(0),self.ucb(1))

        



class MCTS:
    def __init__(self,jogo = None):
    
        if jogo == None:
            jogo = s.Surround()
            jogo.reset()
        self.jogo= jogo
        self.root = Node((0,0),None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollout = 0

    def select_node(self):
        node = self.root
        jogo = 
