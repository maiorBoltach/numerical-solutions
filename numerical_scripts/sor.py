# Метод верхней релаксации
import time
from math import sqrt

import numpy as np


# acc - точность, it_max - макс. число итераций
def sor(A, b, x0, o, acc, it_max):  # A - матрица, b - свободный столбец
    simm = 0  # x0 - начальное приближение
    for i in range(1, len(A), 1):  # Проверка метода на применимость
        for j in range(0, i, 1):
            if A[i, j] == A[j, i]:
                simm += 1
    if simm == (len(A) ** 2 - len(A)) / 2 and 1 < o < 2:
        start_time = time.time()
        x = np.zeros((2, len(A)), float)
        x[1, :] = x0
        k = 0
        while sqrt(sum((x[1] - x[0]) ** 2)) > acc and k < it_max:
            x[0, :] = x[1, :]
            for i in range(0, len(A), 1):
                x[1, i] = o * (b[i] - (sum(A[i, :i] * x[1, :i]) + sum(A[i, i + 1:] * x[0, i + 1:]))) / A[i, i] + (
                        1 - o) * x[0, i]
            k = k + 1
        exec_time = "---%s seconds ---" % (time.time() - start_time)
        if k == it_max:
            print("---Maximum number of iterations is reached!---")
            print("---Solution may not be accurate!---\n\n")
            return x[1], k, exec_time
        else:
            return x[1], k, exec_time
    else:
        print("---Matrix or omega is not correct!---")
        return x0, 0, 0


def sor_vec(A, b, x0, o, acc, it_max):  # Векторная вариация метода
    simm = 0
    for i in range(1, len(A), 1):
        for j in range(0, i, 1):
            if A[i, j] == A[j, i]:
                simm += 1
    if simm == (len(A) ** 2 - len(A)) / 2 and 1 < o < 2:
        start_time = time.time()
        L = np.zeros((len(A), len(A)), float)
        U = np.zeros((len(A), len(A)), float)
        D = np.zeros((len(A), len(A)), float)
        for i in range(0, len(A), 1):
            L[i, :i] = A[i, :i]
            D[i, i] = A[i, i]
            U[i, i + 1:] = A[i, i + 1:]
        x = np.zeros((2, len(A)), float)
        x[1, :] = x0
        k = 0
        while sqrt(sum((x[1] - x[0]) ** 2)) > acc and k < it_max:
            x[0, :] = x[1, :]
            x[1] = np.dot(np.linalg.inv(D + o * L), (np.dot((1 - o) * D - o * U, x[0]) + o * b))
            k = k + 1
        exec_time = "---%s seconds ---" % (time.time() - start_time)
        if k == it_max:
            print("---Maximum number of iterations is reached!---")
            print("---Solution may not be accurate!---\n\n")
            return x[1], k, exec_time
        else:
            return x[1], k, exec_time
    else:
        print("---Matrix or omega is not correct!---")
        return x0, 0, 0
