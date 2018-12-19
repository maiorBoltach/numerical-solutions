# Поиск максимального по модулю собственного значения матрицы
import time

import numpy as np


def pm(A, q0, acc, it_max):
    start_time = time.time()
    q = np.array(q0, float)
    k = 0
    lmbda = np.zeros(2, float)
    lmbda[0] = 1
    while k < it_max and abs(lmbda[1] - lmbda[0]) > acc:
        lmbda[0] = lmbda[1]
        z = np.dot(A, q)
        q = z / np.linalg.norm(z)
        lmbda[1] = np.dot(np.dot(A, q), q)
    exec_time = "---%s seconds ---" % (time.time() - start_time)
    return lmbda[1], exec_time
