# Метод Рунге-Кутта 4-го порядка
from math import sin, pi

import matplotlib.pyplot as plt
import numpy as np


def RK(F, tau, T, y0):
    N = int(T / tau) + 1
    t_mas = np.array([tau * n for n in range(0, N)])
    y = np.zeros((N, 2), float)
    y[0] = y0
    print(t_mas)
    print(y)
    k = 0
    while tau * (k + 1) < T:
        k1 = F(t_mas[k], y[k])
        k11 = [tau * i / 2 for i in k1]
        k2 = F(t_mas[k] + tau / 2, y[k] + k11)
        k22 = [tau * i / 2 for i in k2]
        k3 = F(t_mas[k] + tau / 2, y[k] + k22)
        k33 = [tau * i for i in k3]
        k4 = F(t_mas[k] + tau, y[k] + k33)
        k2 = [2 * i for i in k2]
        k3 = [2 * i for i in k3]
        temp = k1 + k2 + k3 + k4
        K = [tau / 6 * i for i in temp]
        print(K)
        y[k + 1] = y[k] + K
        k += 1
    return t_mas, y


def F(yn):
    return np.array([yn[1], -sin(yn[0])])


t, y = RK(F, pi / 50, 4 * pi, [1, 0])
u = np.array(y[:, 0])
print(u)
plt.plot(t, u, label="RK")
plt.legend()
plt.ylabel("y[t]")
plt.xlabel("x")
plt.show()
