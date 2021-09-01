import numpy as np
import scipy.linalg
from stats import betweenClassCovariance, withinClassCovariance
from dataset import load_db, NUM_CLASSES


def LDA(D, L, K, m):
	S_B = betweenClassCovariance(D, L, K)
	S_W = withinClassCovariance(D, L, K)

	# generalized eigenvalue problem
	s, U = scipy.linalg.eigh(S_B, S_W)
	W = U[:, ::-1][:, 0:m]

	#UW, _, _ = np.linalg.svd(W)
	#U = UW[:, 0:m]

	# project every point to m-dimensional subspace
	DP = np.dot(W.T, D)     # dim = mxN
	return DP


def LDA_all(D, L, K):		# K = number of classes
    listDP = []
    for m in range(1, K):	# compute LDA for every direction m < K
        DP_m = LDA(D, L, K, m)
        listDP.append(DP_m)
    
    return listDP


def main():
    D, L = load_db()

    # LDA: computing and plotting all DP for every m (< K)
    listDP = LDA_all(D, L, NUM_CLASSES)

if __name__ == "__main__":
    main()