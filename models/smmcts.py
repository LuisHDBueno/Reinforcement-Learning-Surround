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
    """SMMCTS Node

    Attributes:
        move: moves taken from parent (move_player1, move_player2)
        parent: parent node
        N: number of simulations
        Q: number of wins for each player
        children: key: move, item: child Node
        outcome: -1 if not terminal, 0 if draw, 1 if p1 wins, 2 if p2 wins
    Methods:
        add_children: add children to tree. There should be children for all legal moves
        ucb1: calculates index based policy UCB1 for given player 
        var: calculates variance of the node for given player
        ucb1_tuned: calculate index based policy UCB1-Tuned for given player
        value: calculates selection value
        get_win_rate: returns the win rate for the given player
        get_root: returns the root node
    """    
    def __init__(self, move:tuple[int],parent):
        """Init of class Node

        :param move: moves taken from parent (move_player1, move_player2)
        :type move: tuple
        :param parent: parent node
        :type parent: Node
        """        
        self.move = move
        self.parent = parent
        self.N = 0 # number of simulations
        self.Q = np.zeros((2,), dtype = np.int32) # number of wins for each player
        self.children = {} # key: move, item: child Node
        self.outcome = -1 # -1 if not terminal, 0 if draw, 1 if p1 wins, 2 if p2 wins

    def add_children(self, children) -> None:
        """Add children to tree. 
        There should be children for all legal moves

        :param children: children moves to be added
        :type children: Iterable[tuple[int]]
        """        
        for child in children:
            self.children[child] = Node(child,self)
        
    def ucb1(self, player:bool) -> float:
        """Calculates index based policy UCB1 for given player 
        :param player: 0 for player 1, 1 for player 2
        :type player: bool
        :return: policy value
        :rtype: float
        """        
        if self.N == 0:
            return np.inf
        return self.Q[player]/self.N + np.sqrt(2*(np.log(self.parent.N))/self.N)
    
    def var(self, player:bool) -> float:
        """Calculates variance of the node for given player

        :param player: 0 for player 1, 1 for player 2
        :type player: bool
        :return: variance
        :rtype: float
        """        
        if self.N == 0 or len(self.children) <= 1:
            return np.inf
        m = np.mean([child.Q[player] / child.N for child in self.children.values()])
        Sn = 0

        for child in self.children.values():
            Sn += (child.Q[player] / child.N - m) ** 2
        Sn = Sn / (len(self.children)-1)

        return Sn
    
    def ucb1_tuned(self, player:bool) -> float:
        """Calculates index based policy UCB1-Tuned for given player

        :param player: 0 for player 1, 1 for player 2
        :type player: bool
        :return: policy value
        :rtype: float
        """            
        if self.N == 0:
            return 0
        
        Sn = self.parent.var(player)

        return self.Q[player] / self.N + np.sqrt(min(0.25,Sn) / self.N)
       
    def value(self, use_ucb1:bool =True) -> tuple[float]:
        """Calculates selection value

        :param use_ucb1: if True select child using UCB1, otherwise select child using UCB1-Tuned, defaults to True
        :type use_ucb1: bool, optional
        :return: selection value of each player
        :rtype: tuple(float,float)
        """        
        if use_ucb1:
            return (self.ucb1(0), self.ucb1(1))
        
        return (self.ucb1_tuned(0), self.ucb1_tuned(1))
    
    def get_win_rate(self, player:bool) -> float:
        """Returns the win rate for the given player

        :param player: 0 for player 1, 1 for player 2
        :type player: bool
        :return: win rate
        :rtype: float
        """        
        if self.N == 0:
            return 0
        
        return self.Q[player] / self.N
    
    def get_root(self):
        """Returns the root node

        :return: root node
        :rtype: Node
        """        
        node = self

        while node.parent is not None:
            node = node.parent

        return node

class SMMCTS:
    """Simultaneous Move Monte Carlo Tree Search
    
    Attributes:
        root_game: root game
        root: root node
        node_count: number of nodes in the tree
        num_rollout: number of rollouts
        curr_node: current node
        curr_game: current game
        use_ucb1: if True select child using UCB1, otherwise select child using UCB1-Tuned
    Methods:
        legal_moves: returns a list of legal moves for both players
        game_over: chech if game is over
        expand: expand the tree adding all possible actions from the current state
        select_node: select a node to expand
        select_node_curr: select a node to expand from curr_game
        roll_out: roll out the game until the end
        back_propagate: back propagate the outcome of the game
        search: grow tree
        time_search: grow tree until time limit
        search_curr: grow tree
        move: move the curr to the child corresponding to the move
        best_move: returns the best move for each player for curr_game
        best_move_timer: returns the best move for each player for curr_game
        get_buffers: get the buffer of the tree
        __sizeof__: returns the size of the tree
    """
    def __init__(self,use_ucb1:bool = True, game: s.Surround = None):
        """Init of class SMMCTS
        :param use_ucb1: if True select child using UCB1, otherwise select child using UCB1-Tuned, defaults to True
        :type use_ucb1: bool, optional
        :param game: root game, defaults to None
        :type game: s.Surround, optional
        """        
        if game == None:
            game = s.Surround()
            game.reset()
        self.root_game= game
        self.root = Node((0,0),None)
        self.node_count = 0
        self.num_rollout = 0
        self.curr_node = self.root
        self.curr_game = self.root_game
        self.use_ucb1 = use_ucb1

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
        moves = []
        
        if not board[player1[0]+1, player1[1]]: moves1.append(1)
        if not board[player1[0], player1[1]+1]: moves1.append(2)
        if not board[player1[0]-1, player1[1]]: moves1.append(3)
        if not board[player1[0], player1[1]-1]: moves1.append(4)

        if not board[player2[0]+1, player2[1]]: moves2.append(1)
        if not board[player2[0], player2[1]+1]: moves2.append(2)
        if not board[player2[0]-1, player2[1]]: moves2.append(3)
        if not board[player2[0], player2[1]-1]: moves2.append(4)

        for move1 in moves1:
            for move2 in moves2:
                    moves.append((move1,move2))
        
        if len(moves1) == 0:
            moves = [(0, move2) for move2 in moves2]
        elif len(moves2) == 0:
            moves = [(move1, 0) for move1 in moves1]
        
        return moves
    
    def game_over(self, game: s.Surround) -> bool:
        """Checks if game is over

        :param game: game state
        :type game: s.Surround
        :return: True if game is over, False otherwise
        :rtype: bool
        """        
        lose1, lose2 = game.lose1, game.lose2

        if lose1 or lose2:
            return True
        
        return False
    
    def expand(self, parent: Node, game: s.Surround) -> bool:
        """Expand the tree adding all possible actions from the current state

        :param parent: parent node
        :type parent: Node
        :param game: game state
        :type game: s.Surround
        :return: True if the game is not over, False otherwise
        :rtype: bool
        """        
        if self.game_over(game):
            return False
        if len(self.legal_moves(game)) == 0:
            return False
        
        moves = self.legal_moves(game)
        parent.add_children(moves)
        self.node_count += len(moves)

        return True
    
    def select_node(self) -> tuple[Node,s.Surround]:
        """Select a node to expand
        
        :return: node to expand, game state
        :rtype: tuple[Node,s.Surround]
        """        
        node = self.root
        game = deepcopy(self.root_game)

        while len(node.children):
            children = node.children.values()
            
            max_value1 = max(children, key=lambda child: child.value(self.use_ucb1)[0]).value(self.use_ucb1)[0]
            max_value2 = max(children, key=lambda child: child.value(self.use_ucb1)[1]).value(self.use_ucb1)[1]
            max_nodes1 = [child for child in children if child.value(self.use_ucb1)[0] == max_value1]
            max_nodes2 = [child for child in children if child.value(self.use_ucb1)[1] == max_value2]
            node1 = random.choice(max_nodes1).move[0]
            node2 = random.choice(max_nodes2).move[1]
            node = node.children[(node1,node2)]
            game.step(node.move)
        
        self.expand(node, game)
        return node, game
        
    def select_node_curr(self) -> tuple[Node,s.Surround]:
        """Select a node to expand from curr_game

        
        :return: node to expand, game state
        :rtype: tuple[Node,s.Surround]
        """        
        node = self.curr_node
        game = deepcopy(self.curr_game)

        while len(node.children):
            children = node.children.values()
            max_value1 = max(children, key=lambda child: child.value(self.use_ucb1)[0]).value(self.use_ucb1)[0]
            max_value2 = max(children, key=lambda child: child.value(self.use_ucb1)[1]).value(self.use_ucb1)[1]
            max_nodes1 = [child for child in children if child.value(self.use_ucb1)[0] == max_value1]
            max_nodes2 = [child for child in children if child.value(self.use_ucb1)[1] == max_value2]
            node1 = random.choice(max_nodes1).move[0]
            node2 = random.choice(max_nodes2).move[1]
            node = node.children[(node1,node2)]
            game.step(node.move)
        
        self.expand(node, game)
        return node, game
        
    def roll_out(self, game: s.Surround) -> int:
        """Roll out the game until the end

        :param game: game state
        :type game: s.Surround
        :return: outcome of the game
        :rtype: int
        """        
        lose1, lose2, = False, False
        while not lose1 and not lose2:
            moves = self.legal_moves(game)
            if len(moves):
                _,_,_,lose1,lose2, _ = game.step(random.choice(moves))
            else:
                lose1, lose2 = True, True

        if lose1 and lose2:
            return [0,0]
        if lose1:
            return [-1,1]
        if lose2:
            return [1,-1]
    
    def roll_out2(self, game: s.Surround) -> int:
        """Roll out the game until the end
        
        :param game: game state
        :type game: s.Surround
        :return: outcome of the game
        :rtype: int
        """
        lose1, lose2, = False, False
        old_moves = []

        while not lose1 and not lose2:
            moves = self.legal_moves(game)
            moves1 = [move[0] for move in moves]
            moves2 = [move[1] for move in moves]

            if len(moves):
                if len(old_moves):
                    if old_moves[0] in moves1:
                        p = random.random()
                        if p < 0.68:
                            move1 = old_moves[0]
                        else:
                            move1 = random.choice(moves1)
                    if old_moves[1] in moves2:
                        p = random.random()
                        if p < 0.68:
                            move2 = old_moves[1]
                        else:
                            move2 = random.choice(moves2)
                else:
                    move1 = random.choice(moves1)
                    move2 = random.choice(moves2)
                _,_,_,lose1,lose2,_ = game.step((move1,move2))
                old_moves = [move1,move2]
            else:
                lose1, lose2 = True, True

        if lose1 and lose2:
            return [0,0]
        if lose1:
            return [-1,1]
        if lose2:
            return [1,-1]
        
    def back_propagate(self, node: Node, outcome: tuple[int]) -> None:
        """Back propagate the outcome of the game

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

        :param count: number of times to perform tree growth, defaults to 1000
        :type count: int, optional
        :return: best move, time used
        :rtype: tuple[tuple[int],float]
        """        
        for _ in range(count):
            node, game = self.select_node()
            if len(node.children):
                for child in node.children.values():
                    game2 = deepcopy(game)
                    game2.step(child.move)
                    outcome = self.roll_out2(game2)
                    self.back_propagate(child, outcome)
                     
    def time_search(self, time_limit: float = 1.0):
        """Grow tree until time limit

        :param time_limit: time limit, defaults to 1.0
        :type time: float, optional
        """        
        start_time = time.time()
        while time.time() - start_time < time_limit:
            self.search(1)
    
    def search_curr(self, count: int = 50) -> None:
        """Grow tree

        :param count: number of times to perform tree growth, defaults to 50
        :type count: int, optional
        :return: best move, time used
        :rtype: tuple[tuple[int],float]
        """        
        for _ in range(count):
            node, game = self.select_node_curr()
            if len(node.children):
                for child in node.children.values():
                    game2 = deepcopy(game)
                    game2.step(child.move)
                    outcome = self.roll_out2(game2)
                    self.back_propagate(child, outcome)
    
    def move(self, move: tuple[int]) -> None:
        """Move the curr to the child corresponding to the move

        :param move: move to be performed
        :type move: tuple[int]
        """       
        if self.game_over(self.curr_game):
            self.curr_game.reset()
            self.curr_node = self.root
        if move in self.curr_node.children:
            self.curr_game.step(move)
            self.curr_node = self.curr_node.children[move]
        else:
            self.curr_game.step(move)
            if not self.game_over(self.curr_game):
                self.search_curr()
        if self.game_over(self.curr_game):
            self.curr_game.reset()
            self.curr_node = self.root

    def best_move(self,num_search = 50) -> tuple[int,int]:
        """Returns the best move for each player for curr_game

        :return: best move for each player
        :rtype: tuple[int,int]
        """        
        if self.game_over(self.curr_game):
            return (1,1)
        
        self.search_curr(num_search)
        if self.curr_node.children == {}:
            return (1, 1)
        
        max_value1 = max(self.curr_node.children.values(), key=lambda child: child.get_win_rate(0)).get_win_rate(0)
        max_value2 = max(self.curr_node.children.values(), key=lambda child: child.get_win_rate(1)).get_win_rate(1)
        
        if max_value1 == 0:
            max_nodes1 = [child for child in self.curr_node.children.values()]
        else:
            max_nodes1 = [child for child in self.curr_node.children.values() if child.Q[0]/child.N == max_value1]
        if max_value2 == 0:
            max_nodes2 = [child for child in self.curr_node.children.values()]
        else:
            max_nodes2 = [child for child in self.curr_node.children.values() if child.Q[1]/child.N == max_value2]
        
        best_child1 = random.choice(max_nodes1).move[0]
        best_child2 = random.choice(max_nodes2).move[1]
        
        return (best_child1, best_child2)
    
    def best_move_timer(self, time_limit: float = 1) -> tuple[int,int]:
        """Returns the best move for each player for curr_game

        :param time_limit: time to make decision, defaults to 1
        :type time_limit: float, optional
        :return: move for each player
        :rtype: tuple[int,int]
        """
        if self.game_over(self.curr_game):
            return (1,1)
        
        start = time.time()

        while time.time() - start < time_limit:
            self.search_curr(1)

        if self.curr_node.children == {}:
            print("no children")
            return (1, 1)
        
        max_value1 = max(self.curr_node.children.values(), key=lambda child: child.get_win_rate(0)).get_win_rate(0)
        max_value2 = max(self.curr_node.children.values(), key=lambda child: child.get_win_rate(1)).get_win_rate(1)
        
        if max_value1 == 0:
            max_nodes1 = [child for child in self.curr_node.children.values()]
        else:
            max_nodes1 = [child for child in self.curr_node.children.values() if child.Q[0]/child.N == max_value1]
        
        if max_value2 == 0:
            max_nodes2 = [child for child in self.curr_node.children.values()]
        else:
            max_nodes2 = [child for child in self.curr_node.children.values() if child.Q[1]/child.N == max_value2]
        
        best_child1 = random.choice(max_nodes1).move[0]
        best_child2 = random.choice(max_nodes2).move[1]
        
        return (best_child1, best_child2)
    
    def get_buffers(self) -> tuple[np.array,np.array]:
        """Get the buffer of the tree

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
        queue.put((node, board))

        while not queue.empty():
            node, board = queue.get()
            boards_buffer.append(board)
            probs = np.zeros((4,), dtype=np.float32)
            player1 = np.nonzero(board[:,:,1])
            player1 = player1[0][0], player1[1][0]
            player2 = np.nonzero(board[:,:,2])
            player2 = player2[0][0], player2[1][0]

            for child in node.children.values():
                child_board = np.zeros((s.BOARD_WIDTH, s.BOARD_HEIGHT, 3), dtype=np.bool)
                child_board[:,:,0] = board[:,:,0]
                board[player1[0]+child.move[0][0], player1[1]+child.move[1][1], 1] = 1
                board[player2[0]+child.move[0][0], player2[1]+child.move[1][1], 2] = 1
                board[player1[0], player1[1], 0] = 1
                board[player2[0], player2[1], 0] = 1
                probs[child.move[0]-1] += child.value(self.use_ucb1)[0]
                queue.put((child, child_board))

            probs_buffer.append(probs)

        return np.array(boards_buffer), np.array(probs_buffer)
    
    def __sizeof__(self) -> int:
        """Returns the size of the tree

        :return: size of the tree
        :rtype: int
        """        
        return self.node_count

def mcts_battle(mcts1,mcts2, num_games = 10, render = False,frame_rate = 5, timed = False, num_searches =100) -> list[tuple[int,int]]:
    """Watch a battle between two SMMCTS
    
    :param mcts1: SMMCTS 1
    :type mcts1: SMMCTS
    :param mcts2: SMMCTS 2
    :type mcts2: SMMCTS
    :param num_games: number of games to be played, defaults to 10
    :type num_games: int, optional
    :param render: if True renders the game, defaults to False
    :type render: bool, optional
    :param frame_rate: frame rate of the game, defaults to 5
    :type frame_rate: int, optional
    :param timed: if True the SMMCTS will be trained in the time between frames, defaults to False
    :type timed: bool, optional
    :param num_searches: number of searches to be performed, defaults to 100
    :type num_searches: int, optional
    :return: list of tuples with the outcome of each game"""
    jogo = s.Surround(human_render=render, human_controls=0, frame_rate=frame_rate)
    jogo.reset()
    outcomes = []
    
    i = 0
    while i < num_games:
        if timed:
            moves1 = mcts1.best_move_timer(1/frame_rate)
            moves2 = mcts2.best_move_timer(1/frame_rate)
        else:
            moves1 = mcts1.best_move(num_searches)
            moves2 = mcts2.best_move(num_searches)
        moves = (moves1[0], moves2[1])

        _, _, _, lose1, lose2, action = jogo.step(moves)
        mcts1.move(action)
        mcts2.move(action)

        if lose1 or lose2:
            outcomes.append((lose1,lose2))
            i += 1

    return outcomes