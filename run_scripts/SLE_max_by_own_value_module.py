# Нахожение максимального по модулю собственного значения
import numpy as np

from numerical_scripts import pm


def find_max_by_module(N):
    A = np.zeros((N, N), float)
    q0 = np.zeros(N, float)
    q0[0] = 1
    for i in range(0, N, 1):
        for j in range(0, N, 1):
            A[i][j] = 1 / (i + j + 1)
    lambda_max, t = pm.pm(A, q0, 0.001, 1000)
    print(chr(955), " = ", lambda_max, "\nExecution time: ", t, "\n\n")


