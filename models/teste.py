import numpy as np

# matriz de permutacao 16X16 de todas as linhas e colunas
# para ser usada na funcao de rotacao
permutation_matrix = np.zeros((16, 16))
for i in range(16):
    permutation_matrix[i, 15 - i] = 1

print(permutation_matrix)