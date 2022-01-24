import numpy as np
from scipy.linalg import hilbert
import scipy.linalg.decomp_lu as dl

n = 6
A = 4 * np.eye(n) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
print("A =\n", A)
print("K = ", np.linalg.cond(A, 2))
x = np.ones(n)
print("x = ", x)
b = A @ x
print("b = ", b)
my_x = dl.lu_solve(dl.lu_factor(A), b)
print("my_x = ", my_x)
print("relative error = ", np.linalg.norm(my_x - x, 2) / np.linalg.norm(x, 2))

