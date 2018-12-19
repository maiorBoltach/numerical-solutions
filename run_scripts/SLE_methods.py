# Методы Якоби, Зейделя, верхней релаксации, сопряженных градиентов
import numpy as np

from numerical_scripts import cg, seidel, sor, jacobi


def prepare_data(N, a):
    alpha = a
    A = np.zeros((N, N), float)
    b = np.zeros(N)
    x0 = np.zeros(N)
    A[0, 0] = 2
    A[0, 1] = -1 + alpha
    A[N - 1, N - 1] = 2
    A[N - 1, N - 2] = -1 + alpha
    b[0] = 1 - alpha
    b[N - 1] = 1 + alpha
    x0[0] = 0.7
    x0[N - 1] = 0.8
    for i in range(1, N - 1, 1):
        A[i, i] = 2
        A[i, i + 1] = -1 + alpha
        A[i, i - 1] = -1 + alpha
        b[i] = 0
        x0[i] = 0.1 * i
    return A, b, x0


def check_upper_relaxation_method(N, a, o):
    A, b, x0 = prepare_data(N, a)
    xsor, ksor, tsor = sor.sor(A, b, x0, o, 0.00001, 1000)
    return ksor


def check_upper_relaxation_method_vector(N, a, o):
    A, b, x0 = prepare_data(N, a)
    xsorv, ksorv, tsorv = sor.sor_vec(A, b, x0, o, 0.00001, 1000)
    return ksorv


def check_conjugate_gradient_method(N, a):
    A, b, x0 = prepare_data(N, a)
    xcg, kcg, tcg = cg.cg(A, b, 0.00001, 1000)
    return kcg


def check_Jacobi_method(N, a,):
    A, b, x0 = prepare_data(N, a)
    xj, kj, tj = jacobi.jacobi(A, b, x0, 0.00001, 1000)
    return kj


def check_Jacobi_method_vector(N, a):
    A, b, x0 = prepare_data(N, a)
    xjv, kjv, tjv = jacobi.jacobi_vec(A, b, x0, 0.00001, 1000)
    return kjv


def check_Seidel_method(N, a):
    A, b, x0 = prepare_data(N, a)
    xs, ks, tjs = seidel.seidel(A, b, x0, 0.00001, 1000)
    return ks


def check_Seidel_method_vector(N, a):
    A, b, x0 = prepare_data(N, a)
    xsv, ksv, tsv = seidel.seidel_vec(A, b, x0, 0.00001, 1000)
    return ksv
