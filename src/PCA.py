import numpy as np
from stats import covarianceMatrix
from dataset import load_db

def PCA(D, m):
	# D matrix (dim: n x N)
	n = D.shape[0]

	if(n <= m):
		print("Error: new dimensionality (m) is bigger or equal than original dimension")
		exit()

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