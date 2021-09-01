import numpy as np
from stats import covarianceMatrix, plot_scatter
from dataset import load_db

def PCA(D, m):
	# D matrix (dim: n x N)
	n = D.shape[0]

	if(n <= m):
		print("Error: new dimensionality (m) is bigger or equal than original dimension")
		return

	# computing the covariance matrix C
	C = covarianceMatrix(D)

	# computing eigenvalues/eigenvectors and selecting only the eigenvectors associated to the m (< n) biggest eigenvalues
	# 	- s: eigenvalues vector
	#	- U: eigenvectors matrix
	s, U = np.linalg.eigh(C)     # eigh() sort the eigenvalues in ascending order

	# sorting and selecting only the eigenvectors associated to the m (< n) BIGGEST eigenvalues
	l = s[::-1]
	P = U[:, ::-1][:, 0:m]
	
	# Project every points in m-dimensions 
	DP = np.dot(P.T, D)     # dim = mxN

	return DP

def PCA_all(D):
    n = D.shape[0]	# number of original dimensions
    listDP = []

    for m in range(1, n):	# compute PCA for every dimension m < n
        DP_m = PCA(D, m)
        listDP.append(DP_m)
    
    return listDP


def main():
	D, L = load_db()

	# PCA: computing and plotting all DP for every m (< n)
	listDP = PCA_all(D) # index+1 = number of dimension

if __name__ == "__main__":
    main()