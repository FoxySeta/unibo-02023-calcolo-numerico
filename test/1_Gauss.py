import numpy as np
from scipy.linalg import hilbert
import scipy.linalg.decomp_lu as dl

n = 7
A = hilbert(n)
print("A = ", A)
print("K = ", np.linalg.cond(A, 2))
x = np.ones(n)
print("x = ", x)
b = A @ x
print("b = ", b)
my_x = dl.lu_solve(dl.lu_factor(A), b)
print("my_x = ", my_x)
print("relative error = ", np.linalg.norm(my_x - x, 2) / np.linalg.norm(x, 2))

