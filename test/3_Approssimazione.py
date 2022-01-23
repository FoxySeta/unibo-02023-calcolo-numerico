import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

def f(x):
    return (1 + x) ** 2 - (9 + 2 * x) ** 7

def p(x, alpha):
    res = 0
    for i in range(alpha.size):
        res += alpha[i] * x ** i
    return res

def get_A(x, n):
    A = np.zeros((x.size, n))
    for i in range(x.size):
        for j in range(n):
            A[i,j] = x[i] ** j
    return A

def solve(N, n):
    x = np.linspace(-5, 5, N)
    A = get_A(x, n)
    y = f(x)
    U, s, Vh = svd(A)
    alpha = np.zeros(n)
    for i in range(n):
        alpha += (U[:,i] @ y) * Vh[i,:] / s[i]
    accurate_x = np.linspace(-10, 10, 100)
    accurate_f = f(accurate_x)
    accurate_p = p(accurate_x, alpha)
    print('Errore in 0: ', abs((p(0, alpha) - f(0) / f(0))))
    print('Errore nei nodi: ', np.linalg.norm(y - A @ alpha) / \
        np.linalg.norm(y), 2)
    plt.plot(accurate_x, accurate_f, color='blue')
    plt.plot(accurate_x, accurate_p, color='red')
    plt.plot(x, y, marker='o', color='black')
    plt.show()

# aggiungo 1 a n: includo il termine noto del polinomio
solve(15, 5)
solve(15, 8)
solve(15, 13)
