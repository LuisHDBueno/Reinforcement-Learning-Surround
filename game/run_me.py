import surround as s
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))

from net_models import DenseNet, ConvolutionNet

if __name__ == "__main__":
    parameters = sys.argv[1:]

    if len(parameters) > 1:
        raise Exception("Expected only one parameter, got " + str(len(parameters)))
    elif len(parameters) == 0:
        game = s.Surround(human_render=True, human_controls=2, frame_rate=5)
        game.reset()

        for _ in range(10):
            while True:
                game.step((0, 0))
                if game.lose1 or game.lose2:
                    break
    elif parameters[0] == "challenge":
        pass
    elif parameters[0] == "watch":
        pass
    elif parameters[0] == "human":
        pass
    else:
        raise Exception("Invalid parameter: " + parameters[0])