import surround as s
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

import net_models as nm
import smmcts

if __name__ == "__main__":
    parameters = sys.argv[1:]

    # Wrong number of parameters
    if len(parameters) > 1:
        raise Exception("Expected only one parameter, got " + str(len(parameters)))
    
    # No parameter or human game => human vs human
    elif len(parameters) == 0 or parameters[0] == "human":
        game = s.Surround(human_render=True, human_controls=2, frame_rate=5)
        game.reset()

        while True:
            game.step((0, 0))

    # Human vs random
    elif parameters[0] == "random":
        game = s.Surround(human_render=True, human_controls=1, frame_rate=5)
        game.reset()

        model = nm.NeuralNet()

        while True:
            moves1, moves2 = model.legal_moves(game)

            a = 0

            if len(moves2) > 0:
                a = moves2[np.random.randint(len(moves2))]

            game.step((0, a))

    # Best Neural Net vs First Neural Net
    elif parameters[0] == "evolution":
        game = s.Surround(human_render=True, human_controls=0, frame_rate=30)
        game.reset()

        model1 = nm.ConvolutionNet()
        model1.load("cnn_8")
        model2 = nm.ConvolutionNet()
        model2.load("cnn_0")

        while True:
            moves1, moves2 = model1.legal_moves(game)

            action2 = model2.play(game, player=2)
            action1 = model1.play(game)

            game.step((action1, action2))

    # Human vs best SMMCTS model
    elif parameters[0] == "mcts":
        jogo = s.Surround(human_render=True, human_controls=1, frame_rate=5)
        jogo.reset()

        mcts = smmcts.SMMCTS()
        
        while True:
            moves = mcts.best_move_timer(1/5)

            reward, old_board, board, lose1, lose2, action = jogo.step((0, moves[1]))
            
            action = (jogo.player1.last_action, moves[1])
            
            mcts.move(action)

    # Human vs best Neural Net
    elif parameters[0] == "net":
        game = s.Surround(human_render=True, human_controls=1, frame_rate=10)
        game.reset()

        model = nm.ConvolutionNet()
        model.load("cnn_8")

        while True:
            moves1, moves2 = model.legal_moves(game)

            action2 = model.play(game, player=2)

            game.step((0, action2))

    # Unknown parameter
    else:
        raise Exception("Invalid parameter: " + parameters[0])