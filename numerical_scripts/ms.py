# Метод секущих
from math import sin

import matplotlib.pyplot as plt

func = lambda x: 4 * sin(x) + x

a1, b1 = -10.0, 10.0
acc = 0.1


def ms(a, b, f):
    root1 = a if f(a) == 0 else -999
    root2 = b if f(b) == 0 else -999
    c = a - ((b - a) * f(a)) / (f(b) - f(a))
    while abs(b - a) >= acc:
        c = a - ((b - a) * f(a)) / (f(b) - f(a))
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            a, b = a, c
        else:
            a, b = c, b
    rootmas = []
    rootmas.extend([root1, root2, c])
    return rootmas


roots = ms(a1, b1, func)
k = 0
while k < len(roots):
    if roots[k] == -999:
        del roots[k]
        continue
    k += 1

print('Roots: ', roots)
xmas = list(x * 0.1 for x in range(-100, 1, 1))
x1 = list(x * 0.1 for x in range(0, 101, 1))
xmas.extend(x1)

ymas = list(map(func, xmas))

plt.plot(xmas, ymas)
plt.show()
