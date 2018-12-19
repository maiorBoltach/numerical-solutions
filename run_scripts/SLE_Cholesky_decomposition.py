﻿# Разложение Холецкого
from math import sqrt

import numpy as np


def cholesky(A):
    L = np.zeros((len(A), len(A)), float)
    for i in range(0, len(A), 1):
        temp1 = 0
        temp2 = 0
        for j in range(0, i, 1):
            for k in range(0, j, 1):
                temp1 = temp1 + L[i, k] * L[j, k]
            L[i, j] = (A[i, j] - temp1) / L[j, j]
        for t in range(0, i, 1):
            temp2 = temp2 + L[i, t] * L[i, t]
        L[i, i] = sqrt(A[i, i] - temp2)
    return L


def sol_niztr(C, b):
    y = np.zeros(len(C))
    for i in range(0, len(C), 1):
        temp = 0
        for j in range(0, i, 1):
            temp = temp + C[i, j] * y[j]
        y[i] = (b[i] - temp) / C[i, i]
    return y


def sol_vertr(CT, y):
    x = np.zeros(len(CT), float)
    for i in range(len(CT), 0, -1):
        temp = 0
        for j in range(i, len(CT), 1):
            temp = temp + CT[i - 1, j] * x[j]
        x[i - 1] = (y[i - 1] - temp) / CT[i - 1, i - 1]
    return x
