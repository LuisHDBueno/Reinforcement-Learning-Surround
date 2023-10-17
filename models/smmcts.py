
import numpy as np
from copy import deepcopy
import sys
import os
import random
import time
from queue import Queue
import pickle
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
        if self.N == 0:
            return 0
        return self.Q[player]/self.N + np.sqrt(2*(np.log(self.parent.N))/self.N)

    def value(self) -> tuple[float]:
        """ calculates selection value

        :return: selection value of each player
        :rtype: tuple(float,float)
        """        

        return (self.ucb1(0),self.ucb1(1))
    
    def get_win_rate(self, player:bool) -> float:
        """returns the win rate for the given player

        :param player: 0 for player 1, 1 for player 2
        :type player: bool
        :return: win rate
        :rtype: float
        """        
        if self.N == 0:
            return 0
        return self.Q[player]/self.N
    
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
        self.curr_node = self.root
        self.curr_game = self.root_game

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
        #print("board:", board.astype(np.int8))
        #print("player1:", game.board[:,:,1].astype(np.int8))
        #print("player2:", game.board[:,:,2].astype(np.int8))
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
        return moves
    
    def game_over(self, game: s.Surround) -> bool:
        """chech if game is over

        :param game: game state
        :type game: s.Surround
        :return: True if game is over, False otherwise
        :rtype: bool
        """        
        lose1, lose2,  = game.lose1, game.lose2
        if lose1 or lose2:
            return True
        return False
    
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
        if len(self.legal_moves(game)) == 0:
            return False
        children = [Node(move,parent) for move in self.legal_moves(game)]
        parent.add_children(children)
        self.node_count += len(children)
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
            max_nodes1 = [child for child in children if child.value()[0] == max_value1]
            max_nodes2 = [child for child in children if child.value()[1] == max_value2]
            node1 = random.choice(max_nodes1).move[0]
            node2 = random.choice(max_nodes2).move[1]
            node = node.children[(node1,node2)]
            game.step(node.move)

            if node.N == 0:
                return node, game
        
        if self.expand(node, game):            
            node = random.choice(list(node.children.values()))
            game.step(node.move)
        
        return node, game
    def select_node_curr(self) -> tuple[Node,s.Surround]:
        """select a node to expand from curr_game

        :return: node to expand, game state
        :rtype: tuple[Node,s.Surround]
        """        
        node = self.curr_node
        game = deepcopy(self.curr_game)

        while len(node.children):
            children = node.children.values()
            max_value1 = max(children, key=lambda child: child.value()[0]).value()[0]
            max_value2 = max(children, key=lambda child: child.value()[1]).value()[1]
            max_nodes1 = [child for child in children if child.value()[0] == max_value1]
            max_nodes2 = [child for child in children if child.value()[1] == max_value2]
            node1 = random.choice(max_nodes1).move[0]
            node2 = random.choice(max_nodes2).move[1]
            node = node.children[(node1,node2)]
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
        if lose1 and lose2:
            return [0,0]
        if lose1:
            return [-1,1]
        if lose2:
            return [1,-1]
    
    def roll_out2(self, game: s.Surround) -> int:
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
                _,_,_,lose1,lose2 = game.step((move1,move2))
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
            outcome = self.roll_out2(game)
            self.back_propagate(node, outcome)
            self.num_rollout += 1
        
    def search_curr(self, count: int = 50) -> None:
        """Grow tree

        :param count: number of times to perform tree growth, defaults to 50
        :type count: int, optional
        :return: best move, time used
        :rtype: tuple[tuple[int],float]
        """        
        
        for _ in range(count):
            node, game = self.select_node_curr()
            outcome = self.roll_out2(game)
            self.back_propagate(node, outcome)
            self.num_rollout += 1
    
    
    def move(self, move: tuple[int]) -> None:
        """move the curr to the child corresponding to the move

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
        # raise Exception("Move not in children")

    def best_move(self) -> tuple[int,int]:
        """returns the best move for each player for curr_game

        :return: best move for each player
        :rtype: tuple[int,int]
        """        
        if self.game_over(self.curr_game):
            return None
        if self.curr_node.children == {}:
            if self.legal_moves(self.curr_game) == []:
                return None
            self.search_curr(50)
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
    
    def play(self, num_games: int = 1) -> None:
        """play num_games games

        :param num_games: number of games played, defaults to 1
        :type num_games: int, optional
        """        
        self.curr_game.reset()
        self.curr_node = self.root
        i = 0
        while i<num_games:
            move = self.best_move()
            if move == None:
                print("Game",i,"over")
                print("result: ",self.curr_game.lose1," ", self.curr_game.lose2)
                i+=1
                self.curr_game.reset()
                self.curr_node = self.root
                continue
            if self.game_over(self.curr_game):
                i+=1
                print("Game",i,"over")
                print("result: ",self.curr_game.lose1," ", self.curr_game.lose2)
                self.curr_game.reset()
                self.curr_node = self.root
                continue
            self.move(move)



    
    def get_buffers(self) -> tuple[np.array,np.array]:
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
        queue.put((node,board))
        while not queue.empty():
            node, board = queue.get()
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
                queue.put((child,child_board))
            probs_buffer.append(probs)


        return np.array(boards_buffer), np.array(probs_buffer)
    
    def __sizeof__(self) -> int:
        """returns the size of the tree

        :return: size of the tree
        :rtype: int
        """        
        return self.node_count
    
    def get_game(self,node = None):

        # if node == None:
        #     node = self.root
        
        # moves = []
        # while node.parent is not None:
        #     moves.append(node.move)
        #     node = node.parent
        # moves.reverse()
        
        # game = s.Surround()
        # game.reset()
        # boa
        # for move in moves:
        pass



    
if __name__ == "__main__":
    smmcts = MCTS()
    smmcts.search(10000)


    x = input("Let's play")
    jogo = s.Surround(human_render=True, human_controls=0, frame_rate=10)
    jogo.reset()
    x = 1
    tempo = time.time()
    while x < 10000:
        y = 1
        while y < 100:
            move = (1,1)
            if smmcts.best_move() != None:
                move = smmcts.best_move()
            reward, old_board, board, lose1, lose2 = jogo.step(move)
            smmcts.move(move)
            y += 1
        x += 1
        print(x)
    