# Методы Якоби
import time
from math import sqrt

import numpy as np


# acc - точность, it_max - макс. число итераций
def jacobi(A, b, x0, acc, it_max):  # A - матрица, b - свободный столбец
    t = 0  # x0 - начальное приближение
    for i in range(0, len(A), 1):  # Проверка метода на применимость
        if 2 * abs(A[i, i]) > np.sum(abs(A[i])):
            t += 1
        else:
            break
    if t == len(A):
        start_time = time.time()
        x = np.zeros((2, len(A)), float)
        x[1, :] = x0
        k = 0
        while sqrt(sum((x[1] - x[0]) ** 2)) > acc and k < it_max:
            x[0, :] = x[1, :]
            for i in range(0, len(A), 1):
                x[1, i] = (b[i] - (sum(A[i, :i] * x[0, :i]) + sum(A[i, i + 1:] * x[0, i + 1:]))) / A[i, i]
            k += 1
        exec_time = "---%s seconds ---" % (time.time() - start_time)
        if k == it_max:
            print("---Maximum number of iterations is reached!---")
            print("---Solution may not be accurate!---\n\n")
            return x[1], k, exec_time
        else:
            return x[1], k, exec_time
    else:
        print("---Matrix is not correct!---")
        return x0, 0, 0


def jacobi_vec(A, b, x0, acc, it_max):  # Векторная вариация метода
    t = 0
    for i in range(0, len(A), 1):
        if 2 * abs(A[i, i]) > np.sum(abs(A[i])):
            t += 1
        else:
            break
    if t == len(A):
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
            x[1] = np.dot(np.linalg.inv(D), (np.dot(-(L + U), x[0]) + b))
            k += 1
        exec_time = "---%s seconds ---" % (time.time() - start_time)
        if k == it_max:
            print("---Maximum number of iterations is reached!---")
            print("---Solution may not be accurate!---\n\n")
            return x[1], k, exec_time
        else:
            return x[1], k, exec_time
    else:
        print("---Matrix is not correct!---")
        return x0, 0, 0
