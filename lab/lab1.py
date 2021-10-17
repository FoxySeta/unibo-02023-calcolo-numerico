# Università di Bologna
# Corso di laurea in Informatica
# 02023 - Calcolo numerico
# 29/09/2021
# 
# Stefano Volpe #969766
# lab1.py

import numpy as np
import matplotlib.pyplot as plt

def es1():
    """
    Execute the following code
        import sys
        help(sys.float_info)
        print(sys.float_info)
    and understand the meaning of max, max_exp and max_10_exp.
    """
    import sys
    help(sys.float_info)
    print(sys.float_info)
    # max: maximum representable finite float
    # max_exp: maximum int e such that radix**(e-1) is representable
    # max_10_exp: maximum int e such that 10**e is representable

def es2(epsilon = 1.0):
    """
    Write a code to compute the machine precision ϵ in (float) default
    precision with a WHILE construct. Compute also the mantissa digits number.
    """
    epsilon = 1.0
    mant_dig = 1
    while 1.0 + epsilon / 2.0 > 1.0:
        epsilon /= 2.0
        mant_dig += 1
    print("epsilon = " + str(epsilon))
    print("mant_dig = " + str(mant_dig))

def es3():
    """
    Import NumPy (import numpy as np) and exploit the functions float16 and
    float32 in the while statement and see the differences. Check the result of
    print(np.finfo(float).eps)
    """
    print("float16:")
    epsilon = np.float16(1.0)
    mant_dig = 1
    while np.float16(1.0) + epsilon / np.float16(2.0) > np.float16(1.0):
        epsilon /= np.float16(2.0)
        mant_dig += 1
    print("  epsilon = " + str(epsilon))
    print("  mant_dig = " + str(mant_dig))

    print("float32:")
    epsilon = np.float32(1.0)
    mant_dig = 1
    while np.float32(1.0) + epsilon / np.float32(2.0) > np.float32(1.0):
        epsilon /= np.float32(2.0)
        mant_dig += 1
    print("  epsilon = " + str(epsilon))
    print("  mant_dig = " + str(mant_dig))
    print("np.finfo(float).eps = " + str(np.finfo(float).eps))

def es4():
    """
    Matplotlib is a plotting library for the Python programming language and its
    numerical mathematics extension NumPy, from https://matplotlib.org/
    Create a figure combining together the cosine and sine curves, from 0 to 10:
    - add a legend
    - add a title
    - change the default colors
    """
    linspace = np.linspace(0, 10);
    plt.subplots(constrained_layout = True)[1].secondary_xaxis(0.5);
    plt.plot(linspace, np.sin(linspace), color='blue')
    plt.plot(linspace, np.cos(linspace), color='red')
    plt.legend(['sin', 'cos'])
    plt.title('Sine and cosine from 0 to 10')
    plt.show()

def es5(n):
    """
    Write a script that, given an input number n, computes the numbers of the
    Fibonacci sequence that are less than n.
    """
    if n <= 0:
        return 0
    if n <= 1:
        return 1
    a, b = 0, 1
    cont = 2
    while a + b < n:
        b += a
        a = b - a
        cont += 1
    return cont

def r(k): # assuming k > 0
    if k <= 0:
        return 0
    a, b = 0, 1
    for _ in range(k):
        b += a
        a = b - a
    print(b / a)
    return b / a

def relative_error(k):
    phi = (1.0 + 5 ** 0.5) / 2.0
    return abs(r(k) - phi) / phi

def es6():
    """
    Write a code computing, for a natural number k, the ratio  r(k) = F(k+1) /
    / F(k), where F(k) are the Fibonacci numbers. Verify that, for a large k,
    {{rk}}k converges to the value φ=1+5√2 create a plot of the error (with
    respect to φ)
    """
    arange = np.arange(50)
    plt.plot(arange, [relative_error(i) for i in arange])
    plt.legend(['relative error'])
    plt.show()
