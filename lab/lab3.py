# Università di Bologna
# Corso di laurea in Informatica
# 02023 - Calcolo numerico
# 27/10/2021
# 
# Stefano Volpe #969766
# lab3.py

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.linalg.decomp_lu as LUdec

def es0():
    n = 5 # Grado del polinomio approssimante
    x = np.array([1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3])
    y = np.array([1.18, 1.26, 1.23, 1.37, 1.37, 1.45, 1.42, 1.46, 1.53, 1.59, 1.5])
    print('Shape of x:', x.shape)
    print('Shape of y:', y.shape, "\n")
    N = x.size # Numero dei dati
    A = np.zeros((N, n+1))
    for i in range(n+1):
      A[:, i] = x ** i
    print("A = \n", A)

    ATA = A.T @ A
    ATy = A.T @ y
    lu, piv = LUdec.lu_factor(ATA)
    print('LU = \n', lu)
    print('piv = ', piv)
    alpha_normali = LUdec.lu_solve((lu, piv), ATy)

    U, s, Vh = scipy.linalg.svd(A)
    print('Shape of U:', U.shape)
    print('Shape of s:', s.shape)
    print('Shape of V:', Vh.T.shape)
    alpha_svd = np.zeros(s.shape)
    for i in range(n+1):
        ui = U[ : , i]
        vi = Vh[i, : ]
        alpha_svd = alpha_svd + (ui @ y) * vi / s[i]

    print("Normali: ", alpha_normali)
    print("SVD: ", alpha_svd)
    print("Error: ", np.linalg.norm(alpha_normali - alpha_svd) / np.linalg.norm(alpha_svd))

    def p(alpha, x):
        N = len(x)
        n = len(alpha)
        A = np.zeros((N,n))
        for i in range(n):
            A[:, i] = x ** i
        return A @ alpha

    x_plot = np.linspace(1, 3, 100)
    y_normali = p(alpha_normali, x_plot)
    y_svd = p(alpha_svd, x_plot)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'o')
    plt.plot(x_plot, y_normali, 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Approssimazione tramite Eq. Normali')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'o')
    plt.plot(x_plot, y_svd, 'k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Approssimazione tramite SVD')
    plt.grid()
    plt.show()

def es1():
    
