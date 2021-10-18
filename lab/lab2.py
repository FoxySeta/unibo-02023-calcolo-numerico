# Università di Bologna
# Corso di laurea in Informatica
# 02023 - Calcolo numerico
# 13/10/2021
# 
# Stefano Volpe #969766
# lab2.py

import sys
import matplotlib.pyplot as plt
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
  noto b per il sistema lineare Ax=b.
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
  x = np.array(np.ones(A.shape[0]))
  b = A @ x
  lu, piv = LUdec.lu_factor(A)
  l = np.tril(lu, -1) + np.diag(np.ones(lu.shape[0]))
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

def LTrisol(L, b):
  n = b.size
  x = np.empty(n)
  for i in range(n):
    x[i] = (b[i] - L[i, 0 : i] @ x[0 : i] ) / L[i, i]
  return x

def UTrisol(U, b):
  n = b.size
  x = np.empty(n)
  for i in range(n - 1, -1, -1):
    x[i] = (b[i] - U[i, i + 1 : n] @ x[i + 1 : n] ) / U[i, i]
  return x

def es3():
  """
  Scrivi le due funzioni LTrisol() e UTrisol() per implementare i metodi di
  sostituzione all'avanti e all'indietro, poi:
  - usa la fattorizzazione P,L,U=LUdec.lu(A) sulla matrice degli esercizi
    precedenti;
  - risolvi i sistemi triangolari usando la tue funzioni.
  """
  A = np.array([ [3,-1, 1,-2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1] ])
  x = np.ones(A.shape[0])
  b = A @ x
  P, L, U = LUdec.lu(A)
  inv_P = np.linalg.inv(P)
  print(UTrisol(U, LTrisol(L, inv_P @ b)))

def es4():
  """
  Comprendere i seguenti codici che implementano la fattorizzazione LU senza
  pivoting.
  """
  def LU_fact_NOpiv(A):
    a = np.copy(A)
    n=a.shape[1]
    for k in range(n-1):
      if a[k, k] != 0:
        a[k+1:, k] = a[k+1:, k]/a[k,k]
        a1 = np.expand_dims(a[k+1:, k], 1)
        a2 = np.expand_dims(a[k, k+1:], 0)
        a[k+1:, k+1:] = a[k+1:, k+1:] - (a1 * a2)
    return a

def es5():
  """
  Calcola la fattorizzazione di Choleski sulla matrice A generata come
  A=np.array([[3,−1,1,−2],[0,2,5,−1],[1,0,−7,1],[0,2,1,1]],dtype=np.float)
  A=np.matmul(A,np.transpose(A)) 
  usando la funzione np.linalg.cholesky.
  Verifica la correttezza della fattorizzazione.
  Risolvi il sistema lineare Ax = b dove  x = (1,1,1,1)T.
  """
  A = np.array([ [3, -1, 1, -2], [0, 2, 5, -1], [1, 0, -7, 1], [0, 2, 1, 1] ], \
    np.float)
  A = A @ np.transpose(A)
  L = np.linalg.cholesky(A)
  print("L @ L^T =\n", L, "\n@\n", L.T,  "\n=\n", L @ L.T, "\n=\n", A, "\n= A")
  x = np.ones(A.shape[0])
  b = A @ x
  print(UTrisol(L.T, LTrisol(L, b)))

# Metodi iterativi

def IterativeMethod(A, b, x0, maxit, tol, xTrue, formula):
  n = np.size(x0)
  ite = 0 # number of iterations so far
  x = np.copy(x0) # estimated result
  relErr = np.empty((maxit, 1)) # relative errors for every attempt
  relErr[0] = np.linalg.norm(x0 - xTrue) / np.linalg.norm(xTrue)
  currRelDiff = tol + 1
  while currRelDiff > tol and ite < maxit:
    x_old = np.copy(x)
    for i in range(n):
      formula(A, b, x_old, x, i)
    ite += 1
    relErr[ite] = np.linalg.norm(x - xTrue) / np.linalg.norm(xTrue)
    currRelDiff = np.linalg.norm(x - x_old) / np.linalg.norm(x)
  return x, ite, relErr[:ite]

def JacobiFormula(A, b, x_old, x, i):
  n = b.size
  x[i] = (b[i] - A[i, 0 : i] @ x_old[0 : i] - A[i, i + 1 : n] \
         @ x_old[i + 1 : n]) / A[i, i]

def GaussSeidelFormula(A, b, x_old, x, i):
  n = b.size
  x[i] = (b[i] - A[i, 0 : i] @ x[0 : i] - A[i, i + 1 : n] \
         @ x_old[i + 1 : n]) / A[i, i]

def Jacobi(A, b, x0, maxit, tol, xTrue):
  return IterativeMethod(A, b, x0, maxit, tol, xTrue, JacobiFormula)

def GaussSeidel(A, b, x0, maxit, tol, xTrue):
  return IterativeMethod(A, b, x0, maxit, tol, xTrue, GaussSeidelFormula)

def es6():
  """
  Scrivi le funzioni Jacobi(A, b, x0, maxit, tol, xTrue) e
  GaussSeidel(A, b, x0, maxit, tol, xTrue) per implementare i metodi di Jacobi e
  di Gauss Seidel per la risoluzione di sistemi lineari con matrice a diagonale
  dominante. In particolare:
  - x0  sia l'iterato iniziale;
  - la condizione d'arresto sia dettata dal numero massimo di iterazioni
    consentite maxit e dalla tolleranza tol sulla differenza relativa fra due
    iterati successivi.
  Si preveda in input la soluzione esatta xTrue per calcolare l'errore relativo
  ad ogni iterazione. Entrambe le funzioni restituiscano in output:
  - la soluzione x;
  - il numero  k  di iterazioni effettuate;
  - il vettore  relErr  di tutti gli errori relativi.
  """
  pass

def es7():
  """
  Testa le due funzioni dell'esercizio precedente per risolvere il sistema
  lineare Ax = b dove A è la matrice 10x10
  A=⎛ 5  1  0  0 ... 0 ⎞
    | 1  5  1  0 ... 0 |
    | 0  1  ⋱  ⋱  ⋮  ⋮ |
    | 0  0  ⋱  5  1  0 |
    | 0  0 ... 1  5  1 |
    ⎝ 0  0 ... 0  1  5 ⎠
  e x=(1,1,...,1)T la soluzione esatta.
  Confronta i due metodi e grafica in un unico plot i due vettori relErr.
  """
  n = 10
  A = 5*np.eye(n)+ np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
  xTrue = np.ones(n)
  b = A @ xTrue
  x0 = np.zeros(n)
  maxit = 200
  tol = 1.e-6
  xJacobi, kJacobi, relErrJacobi = Jacobi(A, b, x0, maxit, tol, xTrue) 
  xGS, kGS, relErrGS, = GaussSeidel(A, b, x0, maxit, tol, xTrue) 
  print("Jacobi: ", xJacobi, "\nGaussSeidel: ", xGS)
  fig, ax = plt.subplots()
  rangeJabobi = range (0, kJacobi)
  rangeGS = range(0, kGS)
  ax.plot(rangeJabobi, relErrJacobi, label='Jacobi', color = 'blue')
  ax.plot(rangeGS, relErrGS, label='Gauss Seidel', color = 'red')
  legend = ax.legend()
  plt.xlabel('Iterations')
  plt.ylabel('Relative error')
  plt.title('Iterative methods comparison')
  plt.show()

es7()
