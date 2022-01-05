# Università di Bologna
# Corso di laurea in Informatica
# 02023 - Calcolo numerico
# 25/11/2021
# 
# Stefano Volpe #969766
# lab5.py

import numpy as np
import matplotlib.pyplot as plt
from skimage import metrics
from skimage.data import camera
from numpy import fft
from scipy.optimize import minimize

def es0():
    '''
    Esercizio 1
    - Caricare l'immagine camera() dal modulo skimage.data, rinormalizzandola nel
      range.
    - Applicare un blur di tipo gaussiano con deviazione standard il cui
      kernel ha dimensioni utilizzando le funzioni fornite: gaussian_kernel(),
      psf_fft() ed A().
    - Aggiungere rumore di tipo gaussiano, con deviazione standard, usando la
      funzione np.random.normal().
    - Calcolare il Peak Signal Noise Ratio (PSNR) ed il Mean Squared Error (MSE)
      tra l'immagine degradata e l'immagine originale usando le funzioni
      peak_signal_noise_ratio e mean_squared_error disponibili nel modulo
      skimage.metrics.
    '''
    # Crea un kernel Gaussiano di dimensione kernlen e deviazione standard sigma
    def gaussian_kernel(kernlen, sigma):
        x = np.linspace( - (kernlen // 2), kernlen // 2, kernlen)    
        # Kernel gaussiano unidmensionale
        kern1d = np.exp( - 0.5 * (x ** 2 / sigma))
        # Kernel gaussiano bidimensionale
        kern2d = np.outer(kern1d, kern1d)
        # Normalizzazione
        return kern2d / kern2d.sum()

    # Esegui l'fft del kernel K di dimensione d aggiungendo gli zeri necessari 
    # ad arrivare a dimensione shape
    def psf_fft(kern2d, d, shape):
        # Aggiungi zeri
        K_p = np.zeros(shape)
        K_p[:d, :d] = kern2d

        # Sposta elementi
        p = d // 2
        K_pr = np.roll(np.roll(K_p, - p, 0), - p, 1)

        # Esegui FFT
        K_otf = fft.fft2(K_pr)
        return K_otf

    # Moltiplicazione per A
    def A(x, K):
        x = fft.fft2(x)
        return np.real(fft.ifft2(K * x))

    # Moltiplicazione per A trasposta
    # def AT(x, K):
        # x = fft.fft2(x)
        # return np.real(fft.ifft2(np.conj(K) * x))

    # Immagine in floating point con valori tra 0 e 1
    X = camera().astype(np.float64) / 255.0

    # Genera K, gli autovalori di A
    len = 24
    sigma = 3
    K = psf_fft(gaussian_kernel(len, sigma), len, X.shape)

    # Genera il rumore
    sigma2 = 0.02
    noise = np.random.normal(size = X.shape) * sigma2

    # Aggiungi blur e rumore
    b = A(X, K) + noise
    PSNR = metrics.peak_signal_noise_ratio(X, b)
    # ATb = AT(b, K)

    # Visualizziamo i risultati
    plt.figure(figsize = (30, 10))

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(X, cmap = 'gray', vmin = 0, vmax = 1)
    plt.title('Immagine Originale')

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(b, cmap = 'gray', vmin = 0, vmax = 1)
    #plt.title(f'Immagine Corrotta (PSNR: {PSNR:.2f})')

    plt.show()
    PSNR = metrics.peak_signal_noise_ratio(X, b)
    MSE = MSE = metrics.mean_squared_error(X, b)
    print('This is the MSE: ', MSE)
    print('This is the PSNR: ', PSNR)

def es1():
    '''
    Esercizio 2
    - Importare la function minimize da scipy.optimize e visualizzarne l'help.
    - Usando la function minimize con il metodo CG minimizzare la funzione
    f: R^n -> R definita come:
        f(x) = \\sum_i_n(x_i - 1)^2
    - Analizzare la struttura restituita in output dalla function minimize.
    '''
    def f(X):
        res = (X - np.ones(X.shape)) ** 2
        return np.sum(res)

    def df(X):
        res = 2 * (X-np.ones(X.shape))
        return res

    x0 = np.array([2, -10])
    res = minimize(f, x0, method = 'CG', jac = df)

    print(res)
