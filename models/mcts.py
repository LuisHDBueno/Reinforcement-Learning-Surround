import numpy as np
import net_model as nm

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../game'))

import surround as s

class MCTS():

    def __init__(self) -> None:
        self.game = s.Surround()
        self.encoding = s.Encoding()
        self.visit_count = {}
        self.value = {}
        self.value_avg = {}
        self.probs = {}

    def clear(self) -> None:
        """Clear the MCTS tree.
        """
        self.visit_count = {}
        self.value = {}
        self.value_avg = {}
        self.probs = {}

    def get_actions(self, board: np.array, player_model: nm.ConvulutionNet | nm.DenseNet) -> tuple:
        """Get the best action to play given the current board.

        :param board: Current board
        :type board: np.array
        :param player_model: Player model
        :type player_model: nm.ConvulutionNet | nm.DenseNet
        :return: Best action to play
        :rtype: tuple(int, int)
        """
        best_action1 = player_model.predict(board)
        board[:, :, 1], board[:, :, 2] = board[:, :, 2], board[:, :, 1]
        best_action2 = player_model.predict(board)
        return (best_action1, best_action2)

    def return_board(self, board: np.array, playe1_position:tuple,
                      player2_position:tuple):
        """Return the board to the original state.

        :param board: Current board
        :type board: np.array
        :param playe1_position: Player 1 position
        :type playe1_position: tuple
        :param player2_position: Player 2 position
        :type player2_position: tuple
        """
        self.game.board = board
        self.game.player1.pos_x, self.game.player1.pos_y = playe1_position
        self.game.player2.pos_x, self.game.player2.pos_y = player2_position

    def is_leaf(self, board_bit: int):
        """Check if the node is a leaf.

        :param board_bit: Int representation of the board
        :type board_bit: int
        :return: True if the node is a leaf, False otherwise
        :rtype: bool
        """
        return board_bit not in self.tree
    
    def find_leaf(self, initial_state, enemy_model) -> tuple:
        """_summary_

        :param initial_state: Current state
        :type initial_state: tuple
        :return: (Board, Reward)
        :rtype: tuple
        """
        states_history = []
        actions_history = []
        value = 0
        cur_state = initial_state

        while not self.is_leaf(cur_state):
            states_history.append(cur_state)
            counts = self.visit_count[cur_state]
            total_sqrt = np.sqrt(np.sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # In the firt iteration, we need to add noise for exploration
            if cur_state == initial_state:
                noises = np.random.normal(0, 0.3, probs.shape)
                probs = 0.75 * probs + 0.25 * noises

            score = values_avg + probs * total_sqrt / (1 + counts)

            player1_step = int(np.argmax(score))
            player2_step = self.get_actions(self.encoding.decode(cur_state), enemy_model)[1]
            _, _, board, lose1, lose2 = self.game.step((player1_step, player2_step))
            cur_state = self.encoding.encode(board)
            if lose1 or lose2:
                value = -1

        return (states_history, actions_history, value, cur_state)

    def search_minibatch(self):
        pass
        


"""
    def search_batch(self, count, batch_size, state_int,
                     player, net, device="cpu"):
        for _ in range(count):
            self.search_minibatch(batch_size, state_int,
                                  player, net, device)

    def search_minibatch(self, count, state_int, player,
                         net, device="cpu"):
        
        Perform several MCTS searches.
        
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        planned = set()
        for _ in range(count):
            value, leaf_state, leaf_player, states, actions = \
                self.find_leaf(state_int, player)
            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    leaf_state_lists = game.decode_binary(
                        leaf_state)
                    expand_states.append(leaf_state_lists)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state, states,
                                         actions))

        # do expansion of nodes
        if expand_queue:
            batch_v = model.state_lists_to_batch(
                expand_states, expand_players, device)
            logits_v, values_v = net(batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            values = values_v.data.cpu().numpy()[:, 0]
            probs = probs_v.data.cpu().numpy()

            # create the nodes
            for (leaf_state, states, actions), value, prob in \
                    zip(expand_queue, values, probs):
                self.visit_count[leaf_state] = [0]*game.GAME_COLS
                self.value[leaf_state] = [0.0]*game.GAME_COLS
                self.value_avg[leaf_state] = [0.0]*game.GAME_COLS
                self.probs[leaf_state] = prob
                backup_queue.append((value, states, actions))

        # perform backup of the searches
        for value, states, actions in backup_queue:
            # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
            cur_value = -value
            for state_int, action in zip(states[::-1],
                                         actions[::-1]):
                self.visit_count[state_int][action] += 1
                self.value[state_int][action] += cur_value
                self.value_avg[state_int][action] = \
                    self.value[state_int][action] / \
                    self.visit_count[state_int][action]
                cur_value = -cur_value

    def get_policy_value(self, state_int, tau=1):
        
        Extract policy and action-values by the state
        :param state_int: state of the board
        :return: (probs, values)
        
        counts = self.visit_count[state_int]
        if tau == 0:
            probs = [0.0] * game.GAME_COLS
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count ** (1.0 / tau) for count in counts]
            total = sum(counts)
            probs = [count / total for count in counts]
        values = self.value_avg[state_int]
        return probs, values    
"""
        
        
    # Descer a árvore
    # Cálcular os rewards
    

    