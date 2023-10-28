"""Trainment script for the neural network models."""

import numpy as np
import seaborn as sns
import net_models as nm
from copy import deepcopy

def train_model(model: str, train_iterations: int = 10, min_win_rate: int = 0.2) -> None:
    """Train the model until it reaches a win rate of 0.55 against the adversary.

    :param model: type of model to be trained, either 'dnn' or 'cnn'
    :type model: str
    :param train_iterations: _description_, defaults to 100
    :type train_iterations: int, optional
    :param min_win_rate: Minimum win_rate to stop trainment, defaults to 0.2
    :type min_win_rate: int, optional
    """    
    # Initialize the agents
    if model == 'dnn':
        agent1 = nm.DenseNet()
        agent2 = deepcopy(agent1)
    elif model == 'cnn':
        agent1 = nm.ConvolutionNet()
        agent1.load('cnn_1')
        agent2 = deepcopy(agent1)
    else:
        raise ValueError("Model must be either 'dnn' or 'cnn'.")

    # Initialize the win rate history
    win_rate_history = np.empty([0,])

    for i in range(train_iterations):
        print("Checkpoint 1")
        
        # Train the agent until it reaches the minimum win rate and get win rate history
        train_win_rate_history = agent1.train(adversary=agent2, min_win_rate=min_win_rate)
        win_rate_history = np.append(win_rate_history, train_win_rate_history)

        # Agent 2 is now the best agent
        agent2 = deepcopy(agent1)

        print("Checkpoint 2")

        agent1.save(f'./saved_models/{model}_{i}')

        print("Checkpoint 3")

        # Save win rate history graph
        graph = sns.lineplot(data=win_rate_history)
        graph.set(xlabel='Trainment step', ylabel='Win rate')
        graph.figure.savefig(f'./saved_models/win_rate_history_{model}_{i}.png')

if __name__ == '__main__':
    train_model(model='cnn', train_iterations=10)
    train_model(model='dnn', train_iterations=10)