import surround as s
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

from net_models import DenseNet, ConvolutionNet

if __name__ == "__main__":
    parameters = sys.argv[1:]

    if len(parameters) > 1:
        raise Exception("Expected only one parameter, got " + str(len(parameters)))
    elif len(parameters) == 0 or parameters[0] == "human":
        game = s.Surround(human_render=True, human_controls=2, frame_rate=5)
        game.reset()

        for _ in range(10):
            while True:
                game.step((0, 0))
                if game.lose1 or game.lose2:
                    break

    elif parameters[0] == "challenge":
        game = s.Surround(human_render=True, human_controls=1, frame_rate=5)
        game.reset()

        pass # ATTENTION: This code is not finished yet

    elif parameters[0] == "random":
        game = s.Surround(human_render=True, human_controls=1, frame_rate=5)
        game.reset()

        model = DenseNet()

        for _ in range(10):
            while True:
                moves1, moves2 = model.legal_moves(game)

                a = 0

                if len(moves2) > 0:
                    a = moves2[np.random.randint(len(moves2))]

                game.step((0, a))
                if game.lose1 or game.lose2:
                    break

    else:
        raise Exception("Invalid parameter: " + parameters[0])