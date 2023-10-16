import numpy as np
import seaborn as sns
import net_models as nm
from copy import deepcopy
def train_model(model: str, save_rate: int = 5, train_iterations: int = 100, min_win_rate: int = 0.6) -> None:
    """Train the model until it reaches a win rate of 0.55 against the adversary.

    :param model: _description_
    :type model: str
    :param save_rate: _description_, defaults to 5
    :type save_rate: int, optional
    :param train_iterations: _description_, defaults to 100
    :type train_iterations: int, optional
    :param min_win_rate: Minimum win_rate to stop trainment, defaults to 0.6
    :type min_win_rate: int, optional
    """    
    if model == 'dnn':
        agent1 = nm.DenseNet()
        agent2 = deepcopy(agent1)
    else:
        agent1 = nm.ConvolutionNet()
        agent2 = deepcopy(agent1)

    win_rate_history = np.empty([0,])
    win_rate = agent1.get_win_rate(agent2)
    win_rate_history = np.append(win_rate_history, win_rate)

    for _ in range(train_iterations / save_rate):
        for _ in range(save_rate):
            train_win_rate_history = agent1.train(adversary=agent2, min_win_rate=min_win_rate)
            win_rate_history = np.append(win_rate_history, train_win_rate_history)
            
            win_rate = agent1.get_win_rate(agent2)

            agent2 = agent1.copy()

        agent1.save(f'{model}')

    graph = sns.lineplot(data=win_rate_history)
    graph.set(xlabel='Trainment step', ylabel='Win rate')
    graph.figure.savefig(f'./saved_models/win_rate_history_{model}.png')

if __name__ == '__main__':
    train_model(model='dnn', train_iterations=100)
    train_model(model='cnn', train_iterations=100)