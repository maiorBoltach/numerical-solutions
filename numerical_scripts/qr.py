# QR-разложение
import numpy as np


def R(A):  # Функция находит разложение для одной итерации
    A1 = np.array(A, float)
    N = len(A)
    W = np.zeros((N - 1, N), float)
    for i in range(0, N - 1, 1):
        A0 = A1
        s = (sum(A0[i:, i] ** 2)) ** 0.5
        if (s - A0[i, i]) == 0:
            return np.zeros(N), np.zeros(N)
        mu = (2 * s * (s - A0[i, i])) ** (-0.5)
        W[i, :] = mu * np.array([row[i] for row in A0])
        W[i, i] = mu * (A0[i, i] - s)
        W[i, :i] = 0
        A1 = np.dot((np.identity(N, float) - 2 * np.outer(np.transpose(W[i]), W[i])), A0)

    Q0 = (np.identity(N, float) - 2 * np.outer(np.transpose(W[0]), W[0]))
    for j in range(1, N - 1, 1):
        Q0 = np.dot(Q0, (np.identity(N, float) - 2 * np.outer(np.transpose(W[j]), W[j])))

    return A1, Q0


def qr(A, acc, it_max):  # A - матрица, acc - точность, it_max - максимальное число итераций
    k = 0
    R0, Q0 = R(A)
    R1, Q1 = R(np.dot(R0, Q0))
    while not ((np.diag(R1) - np.diag(R0)) > acc).all() and k < it_max:
        R0, Q0 = R(np.dot(R1, Q1))
        if R0.all() == np.zeros(len(A)).all():
            print(k)
            return R1
        R1, Q1 = R(np.dot(R0, Q0))
        if R1.all() == np.zeros(len(A)).all():
            print(k)
            return R0
        k += 1
    print(k)
    return R1
