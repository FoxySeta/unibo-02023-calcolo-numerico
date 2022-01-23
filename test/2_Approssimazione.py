import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

def f(x):
    return 1 / (1 + 25 * x * x)

def get_A(x, n):
    A = np.empty((x.size, n))
    for column in range(n):
        A[:, column] = x ** column
    return A

def p(x, alpha):
    my_y = 0
    for i, alpha_i in enumerate(alpha):
        my_y += alpha_i * x ** i
    return my_y

def error(x, alpha):
    return abs(f(x) - p(x, alpha))

x_first = -1
x_last = 1
m = 10
n_start = 1
n_stop = 4
accuracy = 100

x = np.linspace(x_first, x_last, m)
x_accurate = np.linspace(x_first, x_last, accuracy)
y = f(x)
for n in range(n_start, n_stop):
    A = get_A(x, n)
    U, s, Vh = svd(A)
    alpha = np.zeros(n)
    for i in range(n):
        alpha += (U[:,i] @ y) * Vh[i,:] / s[i]
    print('m = ', m)
    print('n = ', n)
    print('error in 0: ', error(0, alpha))
    print('error in -0.7: ', error(-0.7, alpha))
    print('error in +0.7: ', error(0.7, alpha))
    plt.plot(x_accurate, f(x_accurate))
    plt.plot(x_accurate, p(x_accurate, alpha))
    plt.show()
