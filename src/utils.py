import numpy as np

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape(1,v.size)

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return DTR, LTR, DTE, LTE


def split_K_folds(D, L, K, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])

    start = 0
    size = int(D.shape[1] / K)
    D_SETS = []
    L_SETS = []
    
    for _ in range(K):
        subset = idx[start : (start+size)]
        D_fold = D[:, subset]
        L_fold = L[subset]
        D_SETS.append(D_fold)
        L_SETS.append(L_fold)
        start += size

    return D_SETS, L_SETS

