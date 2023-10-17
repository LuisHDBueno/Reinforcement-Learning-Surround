import numpy as np

array_probs_buffer = np.array([[0.1, 0.3, 0.4, 0.9],
                               [0.1, 0.3, 1, 0.9],
                               [0.1, 0.3, 0.4, 0.9],
                               [0.9, 0.3, 0.4, 0.1],
                               [0.9, 0.3, 0.4, 0.9],
                               [0, 0, 0, 0]])

print(array_probs_buffer)
print(array_probs_buffer.shape)
print(np.argmax(array_probs_buffer, axis=1))

one_hot_probs = np.zeros((array_probs_buffer.shape[0], 4))
one_hot_probs[np.arange(array_probs_buffer.shape[0]), np.argmax(array_probs_buffer, axis=1)] = 1
array_probs_buffer = one_hot_probs

print(array_probs_buffer)