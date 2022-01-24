import numpy as np
from scipy.linalg.basic import solve_triangular
from scipy.linalg.decomp_cholesky import cholesky

n = 6
A = 4 * np.eye(n) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
print("A =\n", A)
print("K = ", np.linalg.cond(A, 2))
x = np.ones(n)
print("x = ", x)
b = A @ x
print("b = ", b)
U = cholesky(A)
print("cholesky_abs_err_fro = ", np.linalg.norm(A - U.T @ U))

y = solve_triangular(U.T, b, lower = True)
my_x = solve_triangular(U, y)
print("my_x = ", my_x)
print("relative_error_2 = ", np.linalg.norm(my_x - x, 2) / np.linalg.norm(x, 2))

