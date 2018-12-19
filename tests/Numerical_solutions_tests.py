import unittest

import matplotlib.pyplot as plt
import numpy as np

from run_scripts import SLE_Cholesky_decomposition as Lab3
from run_scripts import SLE_Gauss as Lab2
from run_scripts import SLE_methods as Lab4
from run_scripts import SLE_tridiagonal_matrix as Lab1
from run_scripts import SLE_max_by_own_value_module as Lab5

from numerical_scripts import qr


class Test_NumerticalSolutions(unittest.TestCase):

    # Решение системы линейных уравнений с трехдиагональной матрицей
    def test_tridiagonal_matrix(self):
        A = np.array([[160, 2, 0, 0],
                      [6, 185, 5, 0],
                      [0, 3, 193, 11],
                      [0, 0, 8, 134]])
        b = np.array([10, 22, 42, 72])
        x = Lab1.solve_lu3(A, b)
        result = np.dot(A, x)
        print(result)

    # Метод Гаусса с частичным выбором ведущего элемента
    def test_Gauss(self):
        A = np.array([[3, 17, 10],
                      [2, 4, -2],
                      [6, 18, -12]], float)
        x = np.array([1, 3, 4])
        b = np.dot(A, x)
        Lab2.lu_solve(A, b)

    # Разложение Холецкого
    def test_Cholesky_decomposition(self):
        A = np.array([[17, 3, 10],
                      [3, 17, -2],
                      [10, -2, 12]], float)
        x = np.array([1, 3, 4])
        b = np.dot(A, x)
        C = Lab3.cholesky(A)
        print(C)
        y = Lab3.sol_niztr(C, b)
        x = Lab3.sol_vertr(C.transpose(), y)
        print(x)

    # Метод верхней релаксации
    def test_upper_relaxation_method(self):
        a = 0.3
        o = np.arange(1.1, 2, 0.1)
        NO = 5
        ksorn = np.zeros((NO, len(o)))
        for k in range(3, NO + 3, 1):
            for i in range(0, len(o), 1):
                ksorn[k - 3, i] = Lab4.check_upper_relaxation_method(k, a, o[i])

        plt.plot(o, ksorn[0], label="sor_N=3")
        plt.plot(o, ksorn[1], label="sor_N=4")
        plt.plot(o, ksorn[2], label="sor_N=5")
        plt.plot(o, ksorn[3], label="sor_N=6")
        plt.plot(o, ksorn[4], label="sor_N=7")
        plt.legend()
        plt.ylabel("Iterations")
        plt.xlabel("$\\omega$")
        plt.show()

    # Метод верхней релаксации для векторов
    def test_upper_relaxation_method_vector(self):
        a = 0.3
        o = np.arange(1.1, 2, 0.1)
        NO = 5
        ksorvn = np.zeros((NO, len(o)))
        for k in range(3, NO + 3, 1):
            for i in range(0, len(o), 1):
                ksorvn[k - 3, i] = Lab4.check_upper_relaxation_method_vector(k, a, o[i])

        plt.plot(o, ksorvn[0], label="sorvec_N=3")
        plt.plot(o, ksorvn[1], label="sorvec_N=4")
        plt.plot(o, ksorvn[2], label="sorvec_N=5")
        plt.plot(o, ksorvn[3], label="sorvec_N=6")
        plt.plot(o, ksorvn[4], label="sorvec_N=7")
        plt.legend()
        plt.ylabel("Iterations")
        plt.xlabel("$\\omega$")
        plt.show()

    # Метод Якоби и Зейделя для векторов
    def test_Jacobi_and_Seidel_methods_vector_comparison(self):
        al = np.arange(0.1, 1, 0.01)
        kjva = []
        ksva = []
        for i in range(0, len(al), 1):
            kjva.append(Lab4.check_Jacobi_method_vector(10, al[i]))
            ksva.append(Lab4.check_Seidel_method_vector(10, al[i]))
        plt.plot(al, kjva, label="jacobi_vec")
        plt.plot(al, ksva, label="seidel_vec")
        plt.legend()
        plt.ylabel("Iterations")
        plt.xlabel("$\\alpha$")
        plt.show()

    # Метод Якоби и Зейделя
    def test_Jacobi_and_Seidel_methods_comparison(self):
        al = np.arange(0.1, 1, 0.01)
        kja = []
        ksa = []
        for i in range(0, len(al), 1):
            kja.append(Lab4.check_Jacobi_method(10, al[i]))
            ksa.append(Lab4.check_Seidel_method(10, al[i]))
        plt.plot(al, kja, label="jacobi")
        plt.plot(al, ksa, label="seidel")
        plt.legend()
        plt.ylabel("Iterations")
        plt.xlabel("$\\alpha$")
        plt.show()

    # Метод Якоби и Зейделя для векторов (итерации)
    def test_Jacobi_and_Seidel_methods_vector_iteration_comparison(self):
        T = 5
        a = 0.3
        N = []
        kjv = []
        ksv = []
        for i in range(2, T, 1):
            N.append(i + 1)
            kjv.append(Lab4.check_Jacobi_method_vector(i + 1, a))
            ksv.append(Lab4.check_Seidel_method_vector(i + 1, a))
        plt.plot(N, kjv, label="jacobi_vec")
        plt.plot(N, ksv, label="seidel_vec")
        plt.legend()
        plt.ylabel("Iterations")
        plt.xlabel("N")
        plt.show()

    # Методы Якоби, Зейделя и сопряженных градиентов(итерации)
    def test_Jacobi_and_Seidel_methods_iteration_comparison(self):
        T = 5
        a = 0.3
        N = []
        kj = []
        ks = []
        kcg = []
        for i in range(2, T, 1):
            N.append(i + 1)
            kj.append(Lab4.check_Jacobi_method(i + 1, a))
            ks.append(Lab4.check_Seidel_method(i + 1, a))
            kcg.append(Lab4.check_conjugate_gradient_method(i + 1, a))
        plt.plot(N, kj, label="jacobi")
        plt.plot(N, ks, label="seidel")
        plt.plot(N, kcg, label="cg")
        plt.legend()
        plt.ylabel("Iterations")
        plt.xlabel("N")
        plt.show()

    # Нахожение максимального по модулю собственного значения
    def test_max_by_own_value_module(self):
        for k in range(2, 11, 1):
            Lab5.find_max_by_module(k)

    # QR-разложение
    def test_qr_decomposition(self):
        A = np.array([[5, 6, 3], [-1, 0, 1], [1, 2, -1]])
        res = qr.qr(A, 0.00001, 11)
        print(np.diag(res))


if __name__ == '__main__':
    unittest.main()
