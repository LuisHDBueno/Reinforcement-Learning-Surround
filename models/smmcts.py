
import numpy as np
from copy import deepcopy
import sys
import os
import random
import time
from queue import Queue
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
    def __init__(self, move:tuple[int],parent):
        """init of class Node

        :param move: moves taken from parent(move_player1, move_player2)
        :type move: tuple
        :param parent: parent node
        :type parent: Node
        """        
        self.move = move
        self.parent = parent
        self.N = 0 # number of simulations
        self.Q = np.zeros((2,), dtype = np.int32) # number of wins for each player
        self.children = {}#key: move, item: child Node
        self.outcome = -1 #-1 if not terminal, 0 if draw, 1 if p1 wins, 2 if p2 wins

    def add_children(self, children) -> None:
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

        return (self.ucb1(0),self.ucb1(1))
    
    def get_root(self):
        """returns the root node

        :return: root node
        :rtype: Node
        """        
        node = self
        while node.parent is not None:
            node = node.parent
        return node
    

        



class MCTS:
    def __init__(self,game = None):
    
        if game == None:
            game = s.Surround()
            game.reset()
        self.root_game= game
        self.root = Node((0,0),None)
        self.node_count = 0
        self.num_rollout = 0

    def legal_moves(self,game: s.Surround) -> list[tuple[int]]:
        """returns a list of legal moves for both players

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
        if not board[player1[0]+1, player1[1]]: moves1.append(1)
        if not board[player1[0], player1[1]+1]: moves1.append(2)
        if not board[player1[0]-1, player1[1]]: moves1.append(3)
        if not board[player1[0], player1[1]-1]: moves1.append(4)

        if not board[player2[0]+1, player2[1]]: moves1.append(1)
        if not board[player2[0], player2[1]+1]: moves1.append(2)
        if not board[player2[0]-1, player2[1]]: moves1.append(3)
        if not board[player2[0], player2[1]-1]: moves1.append(4)
        
        return [(move1,move2) for move1 in moves1 for move2 in moves2]
    
    def game_over(self, game: s.Surround) -> bool:
        """chech if game is over

        :param game: game state
        :type game: s.Surround
        :return: True if game is over, False otherwise
        :rtype: bool
        """        
        lose1, lose2,  = game.lose1, game.lose2
        if lose1 or lose2:
            return False
        return True
    
    def expand(self, parent: Node, game: s.Surround) -> bool:
        """expand the tree adding all possible actions from the current state

        :param parent: parent node
        :type parent: Node
        :param game: game state
        :type game: s.Surround
        :return: True if the game is not over, False otherwise
        :rtype: bool
        """        
        if self.game_over(game):
            return False
        
        children = [Node(move,parent) for move in self.legal_moves(game)]
        parent.add_children(children)

        return True
    
    def select_node(self) -> tuple[Node,s.Surround]:
        """select a node to expand

        :return: node to expand, game state
        :rtype: tuple[Node,s.Surround]
        """        
        node = self.root
        game = deepcopy(self.root_game)

        while len(node.children):
            children = node.children.values()
            max_value1 = max(children, key=lambda child: child.value()[0]).value()[0]
            max_value2 = max(children, key=lambda child: child.value()[1]).value()[1]
            max_nodes = [child for child in children if child.value() == (max_value1,max_value2)]
            node = random.choice(max_nodes)
            game.step(node.move)

            if node.N == 0:
                return node, game
        
        if self.expand(node, game):
            node = random.choice(list(node.children.values()))
            game.step(node.move)
        
        return node, game

    def roll_out(self, game: s.Surround) -> int:
        """roll out the game until the end

        :param game: game state
        :type game: s.Surround
        :return: outcome of the game
        :rtype: int
        """        
        lose1, lose2, = False, False
        while not lose1 and not lose2:
            moves = self.legal_moves(game)
            if len(moves):
                _,_,_,lose1,lose2 = game.step(random.choice(moves))
            else:
                lose1, lose2 = True, True
        
        return [int(not lose1), int(not lose2)]
    
    def back_propagate(self, node: Node, outcome: tuple[int]) -> None:
        """back propagate the outcome of the game

        :param node: node to back propagate
        :type node: Node
        :param outcome: outcome of the game
        :type outcome: tuple[int]
        """   

        while node is not None:
            node.N += 1
            node.Q += outcome
            node = node.parent
        
    def search(self, count: int = 1000) -> None:
        """Grow tree

        :param count: number of times to perform tree growth, defaults to 50
        :type count: int, optional
        :return: best move, time used
        :rtype: tuple[tuple[int],float]
        """        
        
        for _ in range(count):
            node, game = self.select_node()
            outcome = self.roll_out(game)
            self.back_propagate(node, outcome)
            self.num_rollout += 1
        
    
    
    def move(self, move: tuple[int]) -> None:
        """move the root to the child corresponding to the move

        :param move: move to be performed
        :type move: tuple[int]
        """       
        if move in self.root.children:
            self.root_game.step(move)
            self.root = self.root.children[move]
            return
        
        # self.root_game.step(move)
        # self.root.children[move] = Node(move,self.root)
        # self.root.children[move].N = 1
        # self.root = self.root.children[move]
        raise Exception("Move not in children")
    
    def get_buffer(self) -> tuple[np.array,np.array]:
        """get the buffer of the tree

        :return: boards buffer, probs buffer
        :rtype: tuple[np.array,np.array]
        """        
        boards_buffer = []
        probs_buffer = []
        node = self.root.get_root()
        queue = Queue()
        game = s.Surround()
        game.reset()
        board = game.board
        queue.enqueue((node,board))
        while not queue.is_empty():
            node, board = queue.dequeue()
            boards_buffer.append(board)
            probs = np.zeros((4,), dtype=np.float32)
            player1 = np.nonzero(board[:,:,1])
            player1 = player1[0][0], player1[1][0]
            player2 = np.nonzero(board[:,:,2])
            player2 = player2[0][0], player2[1][0]

            for child in node.children.values():
                child_board = np.zeros((s.BOARD_WIDTH,s.BOARD_HEIGHT,3), dtype=np.bool)
                child_board[:,:,0] = board[:,:,0]
                board[player1[0]+child.move[0][0],player1[1]+child.move[1][1],1] = 1
                board[player2[0]+child.move[0][0],player2[1]+child.move[1][1],2] = 1
                board[player1[0],player1[1],0] = 1
                board[player2[0],player2[1],0] = 1
                probs[child.move[0]-1] += child.value()[0]
                queue.enqueue((child,child_board))
            probs_buffer.append(probs)


        return np.array(boards_buffer), np.array(probs_buffer)
    

