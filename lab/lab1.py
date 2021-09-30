# Università di Bologna
# Corso di laurea in Informatica
# 37635 - Algoritmi e strutture di dati
# 29/09/2021
# 
# Stefano Volpe #969766
# lab1.py

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
    Write a program to compute the machine precision ϵ in (float) default
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
    import numpy as np

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

