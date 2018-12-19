# Метод бисекций
import math

import matplotlib.pyplot as plt

func = lambda x: (1 + x ** 2) * math.exp(-x) + math.sin(x)

a1, b1 = 0.0, 10.0
acc = 0.001


def bic(a, b, f):
    root1 = a if f(a) == 0 else 0
    root2 = b if f(b) == 0 else 0
    roots1 = 0
    roots2 = 0
    x = (a + b) / 2
    while abs(f(x)) >= acc:
        x = (a + b) / 2
        if f(a) * f(x) < 0 and f(b) * f(x) < 0:
            roots1 = bic(a, x, f)
            roots2 = bic(x, b, f)
        elif f(a) * f(x) < 0:
            a, b = a, x
        else:
            a, b = x, b
    rootmas = []
    rootmas.extend([root1, root2, roots1, roots2, (a + b) / 2])
    return rootmas


roots = bic(a1, b1, func)
k = 0
while k < len(roots):
    if roots[k] == 0:
        del roots[k]
        continue
    k += 1

print('Roots: ', roots)
xmas = list(x * 0.1 for x in range(0, 101, 1))

ymas = list(map(func, xmas))

plt.plot(xmas, ymas)
plt.show()
