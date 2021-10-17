# Università di Bologna
# Corso di laurea in Informatica
# 02023 - Calcolo numerico
# 13/10/2021
# 
# Stefano Volpe #969766
# lab2.py

import sys
import numpy as np
import scipy.linalg.decomp_lu as LUdec
from scipy.linalg import hilbert

# Matrici e norme

def es0():
    """
    Considera la matrice A
    A=(1     2     )
      (0.499 1.001 )
    Calcola la norma 1, la norma 2, la norma Frobenius e la norma infinito di A
    con  numpy.linalg.norm() (guarda l'help della funzione).
    Calcola il numero di condizionamento di A con numpy.linalg.cond() (guarda
    l'help della funzione).
    Considera il vettore colonna x=(1,1)T e calcola il corrispondente termine
    noto  b  per il sistema lineare Ax=b.
    Considera ora il vettore b~=(3,1.4985)T e verifica che x~=(2,0.5)T è
    soluzione del sistema Ax~=b~.
    Calcola la norma 2 della perturbazione sui termini noti Δb=∥b−b~∥2 e la
    norma 2 della perturbazione sulle soluzioni Δx=∥x−x~∥2.
    Confronta Δb con Δx.
    """
    A = np.array([[1, 2], [0.499, 1.001]])
    print("Norma-1: ", np.linalg.norm(A, 1));
    print("Norma-2: ", np.linalg.norm(A, 2));
    print("Norma di Frobenius: ", np.linalg.norm(A, 'fro'));
    print("Norma-infinito: ", np.linalg.norm(A, np.inf));
    print("K(A): ", np.linalg.cond(A));
    x = np.array([1, 1])
    b = A @ x
    x2 = np.array([2, 0.5])
    b2 = np.array([3, 1.4985])
    print("A @ x2 = ", A @ x2, " = ", b2, " = b2")
    print("|| Δb || = ", np.linalg.norm(b - b2));
    print("|| Δx || = ", np.linalg.norm(x - x2));

# Metodi diretti

def es1(A = np.array ([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1]
    ], dtype=np.float_)):
    """
    Considera la matrice
    A = ⎛ 3 -1  1 -2⎞
        | 0  2  5 -1|
        | 1  0 -7  1|
        ⎝ 0  2  1  1⎠
 
    Crea il problema test in cui il vettore della soluzione esatta è
    x=(1,1,1,1)T e il vettore termine noto è b=Ax. Guarda l'help del modulo
    scipy.linalg.decomp_lu e usa una delle sue funzioni per calcolare la
    fattorizzazione LU di A con pivolting. Verifica la correttezza dell'output.
    Risolvi il sistema lineare con la funzione lu_solve del modulo decomp_lu
    oppure con scipy.linalg.solve_triangular. Visualizza la soluzione calcolata
    e valutane la correttezza.
    """
    
    print("K(A) = ", np.linalg.cond(A))
    x = np.array(np.ones(np.shape(A)[0]))
    b = A @ x
    lu, piv = LUdec.lu_factor(A)
    l = np.tril(lu, -1) + np.diag(np.ones(np.shape(lu)[0]))
    u = np.triu(lu)
    print("l @ u =\n", l, "\n@\n", u, "\n=\n", l @ u, "\n=\n", A, "\n = A");
    my_x = LUdec.lu_solve((lu, piv), b)
    print("Soluzione calcolata: ", my_x)
    print("Errore relativo: ", np.linalg.norm(abs(x - my_x)) / np.linalg.norm(x))

def es2():
    """
    Ripeti l'esercizio 1 sulla matrice di Hilbert, creata con
    A=scipy.linalg.hilbert(n) per n=5,…,10. In particolare:
    - calcola il numero di condizionamento di A
    - considera il vettore colonna  x=(1,…,1)T
    - calcola il corrispondente termine noto b per il sistema lineare Ax=b e la
      relativa soluzione x~ usando la fattorizzazione LU come nel caso
      precedente.
    """
    for n in range(5, 11):
        print("\nHilbert(", n, ")")
        es1(hilbert(n))

