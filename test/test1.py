import numpy as np
import scipy.linalg.decomp_lu as dl

# Considera la funzione numpy.random.rand e genera una matrice di numeri random
# A, di dimensione 7x7.
n = 7
A = np.random.rand(n, n)

# {{1}} Quanto vale il numero di condizionamento (con norma 2) della matrice? La
# matrice è ben o mal condizionata?
print('K_A = ', np.linalg.cond(A))

# {{2}} Crea il problema test con soluzione esatta x_true = [1, 1, 1, 1, 1, 1,
# 1]^T. Riporta gli elementi del vettore b, termine noto del sistema lineare.
x_true = np.ones(n)
b = A @ x_true
print('b = ', b)

# {{3}} Utilizza le funzioni di scipy.linalg.decomp_lu e fattorizza A con lu(A).
# Risulta necessario permutare le righe di A? Da cosa si evince?
p, l, u = dl.lu(A)
print('Permutation?', not np.array_equal(p, np.eye(n)))

# {{4}} Riporta il valore della norma 'fro' della differenza tra A e la sua
# fattorizzazione calcolata. A cosa è dovuto questo errore?
print('||A - l @ u||_fro = ', np.linalg.norm(A - l @ u, 'fro'))

# {{5}} Usare le funzioni scipy.linalg.solve_triangular e/o scipy.linalg.solve
# per risolvere il sistema lineare sfruttando la fattorizzazione di A. Riporta
# la soluzione ottenuta.
my_x = dl.lu_solve(dl.lu_factor(A), b)
print('my_x = ', my_x)

# {{6}} Calcola la norma 2 della differenza fra la soluzione esatta e la
# soluzione calcolata.
print('||x - my_x||_2 = ', np.linalg.norm(x_true - my_x))
