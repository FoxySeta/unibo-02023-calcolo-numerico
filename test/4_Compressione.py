import matplotlib.pyplot as plt
import numpy as np
from skimage.data import coins
from scipy.linalg import svd

p_start = 1
p_stop = 51

A = coins().astype(np.float64) / 255
m, n = np.shape(A)
# plt.imshow(A, cmap='gray')
# plt.show()

# A_p
ps = np.arange(p_start, p_stop)
rel_error_2 = []
rel_error_f = []
compression_factor = []
U, s, Vh = svd(A)
A_p = np.zeros_like(A, np.float64)

for p in ps:
    A_p += np.outer(U[:,p], Vh[p,:]) * s[p]
    # plt.imshow(A_p, cmap='gray')
    # plt.show()
    rel_error_2.append(np.linalg.norm(A - A_p, 2) / np.linalg.norm(A, 2))
    rel_error_f.append(np.linalg.norm(A - A_p, 'fro') / np.linalg.norm(A, 'fro'))
    compression_factor.append(min(A.shape) / p - 1)

plt.title("Errore relativo in norma 2 di A_p")
plt.plot(ps, rel_error_2, color="red")
plt.show()
plt.title("Errore relativo in norma Frobenius di A_p")
plt.plot(ps, rel_error_f, color="orange")
plt.show()

# Fattore di compressione
plt.title("Fattore di compressione")
plt.plot(ps, compression_factor, color="green")
plt.show()

# A~_p
ps = np.arange(min(m, n) - p_stop + 1, min(m, n))
rel_error_2 = []
rel_error_f = []
A_p = np.zeros_like(A, np.float64)

for p in reversed(ps):
    A_p += np.outer(U[:,p], Vh[p,:]) * s[p]
    # plt.imshow(A_p, cmap='gray')
    # plt.show()
    rel_error_2.insert(0, np.linalg.norm(A - A_p, 2) / np.linalg.norm(A, 2))
    rel_error_f.insert(0, np.linalg.norm(A - A_p, 'fro') / np.linalg.norm(A, 'fro'))

plt.title("Errore relativo in norma 2 di A~_p")
plt.plot(ps, rel_error_2, color="red")
plt.show()
plt.title("Errore relativo in norma Frobenius di A~_p")
plt.plot(ps, rel_error_f, color="orange")
plt.show()
