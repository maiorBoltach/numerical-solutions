# Метод сопряженных градиентов
import time

import numpy as np


# tol - точность, it_max - макс. число итераций
def cg(A, b, tol, it_max):  # A - матрица, b - свободный столбец
    simm = 0  # x0 - начальное приближение
    for i in range(1, len(A), 1):  # Проверка метода на применимость
        for j in range(0, i, 1):
            if A[i, j] == A[j, i]:
                simm += 1
    if simm == (len(A) ** 2 - len(A)) / 2:
        start_time = time.time()
        it = 0
        x = 0
        r = np.copy(b)
        r_prev = np.copy(b)
        rho = np.dot(r, r)
        p = np.copy(r)
        while np.sqrt(rho) > tol * np.sqrt(np.dot(b, b)) and it < it_max:
            it += 1
            if it == 1:
                p[:] = r[:]
            else:
                beta = np.dot(r, r) / np.dot(r_prev, r_prev)
                p = r + beta * p
                w = np.dot(A, p)
                alpha = np.dot(r, r) / np.dot(p, w)
                x = x + alpha * p
                r_prev[:] = r[:]
                r = r - alpha * w
                rho = np.dot(r, r)
        exec_time = "---%s seconds ---" % (time.time() - start_time)
        if it == it_max:
            print("---Maximum number of iterations is reached!---")
            print("---Solution may not be accurate!---\n\n")
            return x[1], it, exec_time
        else:
            return x[1], it, exec_time
    else:
        print("---Matrix is not correct!---")
        return 0, 0, 0
