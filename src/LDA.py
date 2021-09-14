import numpy as np
import scipy.linalg
from stats import betweenClassCovariance, withinClassCovariance
from dataset import load_db, NUM_CLASSES

'''
LDA:
	K = Number of classes
	m = Number of attributes (directions)
'''

def LDA(D, L, K, m):
    if m > K-1:
        exit(-1)

    S_B = betweenClassCovariance(D, L, K)
    S_W = withinClassCovariance(D, L, K)

    # Generalized eigenvalue problem
    s, U = scipy.linalg.eigh(S_B, S_W)
    W = U[:, ::-1][:, 0:m]

    # Find orthogonal basis of W
    UW, _, _ = np.linalg.svd(W)
    U = UW[:, 0:m]

    # Project every point to m-dimensional subspace
    DP = np.dot(U.T, D)     # dim = mxN

    return DP
