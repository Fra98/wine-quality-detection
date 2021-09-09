import numpy as np
from train import TRAINModel

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


def split_K_folds(D, L, K, seed=0, shuffle=False):
    if shuffle:
        np.random.seed(seed)
        idx = np.random.permutation(D.shape[1])
    else:
        idx = np.arange(D.shape[1])

    start = 0
    size = int(D.shape[1] / K)
    D_SETS = []
    L_SETS = []
    
    for i in range(K):
        end = start + size

        if i == K-1 and end != D.shape[1]:  # extend last set if index not reach end
            end = D.shape[1]
        
        subset = idx[start : end]
        D_fold = D[:, subset]
        L_fold = L[subset]
        D_SETS.append(D_fold)
        L_SETS.append(L_fold)
        start += size

    return D_SETS, L_SETS


def K_fold_cross_validation(D, L, K, modelName):
    D_SETS, L_SETS = split_K_folds(D, L, K)

    total_correct = 0

    for i in range(K):
        DTR = np.concatenate(D_SETS[:i] + D_SETS[i+1:], axis=1)
        LTR = np.concatenate(L_SETS[:i] + L_SETS[i+1:], axis=0)        
        DTE = D_SETS[i]
        LTE = L_SETS[i]
        
        ## TRAINING: compute labels for evaluation test
        TM = TRAINModel(DTR, LTR, DTE, modelName)
        CLTE = TM.run()

        ## EVALUATION:
        EVAL = np.equal(CLTE, LTE)            # array of booleans -> DTE_labels[i] == LTE[i]
        num_correct = np.sum(EVAL)
        
        # update total
        total_correct += num_correct

    # Calculating global accuracy
    accuracy = total_correct / L.size
    error_rate = 1 - accuracy

    print("Number of correct prediction:", total_correct, "on a total of", L.size, "elements")
    print("Error rate:", error_rate*100, "%")

    return error_rate


def leave_one_out(D, L, modelName):
    K = L.size
    error_rate = K_fold_cross_validation(D, L, K, modelName)

    return error_rate


def computeErrorRate(CL, L):
    EVAL = np.equal(CL, L)            # array of booleans -> CL[i] == L[i]
    num_correct = np.sum(EVAL)
    accuracy = num_correct / CL.size
    error_rate = 1 - accuracy

    print("Number of correct prediction:", num_correct, "on a total of", CL.size, "elements")
    print("Error rate percentage:", error_rate*100, "%")

    return error_rate
