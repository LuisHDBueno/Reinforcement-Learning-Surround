import numpy as np
import os
import sys
import net_models as nm
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
history = np.zeros((10, 100))


with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

np.savetxt("foo.csv", history.astype(np.int8), delimiter=",")