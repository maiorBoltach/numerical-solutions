# Метод Ньютона
import numpy as np


def F(x):
    y = np.zeros_like(x)
    y[0] = (3 + 2 * x[0]) * x[0] - 2 * x[1] - 3
    y[1:-1] = (3 + 2 * x[1:-1]) * x[1:-1] - x[:-2] - 2 * x[2:] - 2
    y[-1] = (3 + 2 * x[-1]) * x[-1] - x[-2] - 4
    return y


def J(x):
    n = len(x)
    jac = np.zeros((n, n))
    jac[0, 0] = 3 + 4 * x[0]
    jac[0, 1] = -2
    for i in range(n - 1):
        jac[i, i - 1] = -1
        jac[i, i] = 3 + 4 * x[i]
        jac[i, i + 1] = -2
    jac[-1, -2] = -1
    jac[-1, -1] = 3 + 4 * x[-1]
    return jac


def Newton_system(F, J, guess):
    N = len(guess)
    delta = np.ones(N)
    x = np.array(guess, float)
    acc = 0.001
    k = 0
    while max(abs(delta)) > acc and k < 100:
        delta = np.linalg.solve(J(x), -F(x))
        x = x + delta
        k += 1
    return x, k


n = 10
guess = 3 * np.ones(n)
sol, its = Newton_system(F, J, guess)

if its > 0:
    print("x = {}".format(sol))
else:
    print("Решение не найдено!")
