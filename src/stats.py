import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils import mrow, mcol
from plot import plot_config, plot_histograms, plot_scatter, plot_pearsonCorrelationMatrix
from dataset import load_db


def center_data(D):
    mu = D.mean(1)  # mean over all columns, for each attribute
    return D - mu.reshape(D.shape[0], 1)  # remove the mean from all points

def GAU_mu_ML(D):
    return np.mean(D, axis=1)

def GAU_var_ML(D):
    return np.var(D, axis=1)

def covarianceMatrix(D):
    N = D.shape[1] 	# number of samples
    # computing the mean for every row (attributes) and subtract it in all the dataset
    mu = D.mean(1)
    DC = D - mu.reshape(mu.size, 1)

    C = (1/N) * np.dot(DC, DC.T)

    return C

def betweenClassCovariance(D, L, K):
    N = D.shape[1]  # total number of samples
    n = D.shape[0]  # number of dimensions

    mu = D.mean(1)
    mu = mcol(mu)

    S_B = np.zeros((n, n))
    for c in range(K):
        D_c = D[:, L == c]              # samples of class c
        n_c = D_c.shape[1]              # number of samples of class c
        mu_c = D_c.mean(axis=1)         # mean of the class
        mu_c = mcol(mu_c)
        S_B += n_c * np.dot(mu_c-mu, (mu_c-mu).T)
    S_B = (1/N) * S_B

    return S_B


def withinClassCovariance(D, L, K):
    N = D.shape[1]  # total number of samples
    n = D.shape[0]  # number of dimensions

    S_W = np.zeros((n, n))
    for c in range(K):
        D_c = D[:, L==c]                # samples of class c
        n_c = D_c.shape[1]              # number of samples of class c
        C_c = covarianceMatrix(D_c)     # covariance matrix of class c
        S_W += n_c * C_c
    S_W = (1/N) * S_W 

    return S_W

def pearsonCorrelationMatrix(D):
    return np.corrcoef(D)

def gauss_data(D):            
    DG = np.copy(D)    
    for i in range(D.shape[0]):
        for x in range(D.shape[1]):
            summ = 1*(D[i]>D[i][x])            
            DG[i][x] = (summ.sum()+1.0)/(D.shape[1]+1.0)            
    DG = norm.ppf(DG)       
    return DG


def main():
    # LOADING DATABASE
    D, L = load_db()

    # UI config
    plot_config()
    
    plot_histograms(D, L)
    # plot_scatter(D, L, 5, 3)

    # Class distribution
    N_1 = len(L[L==1])
    N_0 = len(L[L==0])
    print(f"Number of sample of class 1 (HQ): {N_1} ({100* N_1/L.size : .2f} %)")
    print(f"Number of sample of class 0 (LQ): {N_0} ({100* N_0/L.size : .2f} %)")

    # Mean over all columns, for each attribute
    mu = D.mean(1)       

    # Data centered around mean
    DC = center_data(D)

    # Covariance matrix
    C = covarianceMatrix(D)

    # Pearson Correlation Coefficient Matrix
    PCC = pearsonCorrelationMatrix(D)
    plot_pearsonCorrelationMatrix(PCC)

    # Gaussianization
    DG = gauss_data(D)
    PCC = pearsonCorrelationMatrix(DG)
    plot_pearsonCorrelationMatrix(PCC)
    plot_histograms(DG, L, prefix='gaussianized')



if __name__ == "__main__":
    main()
    


