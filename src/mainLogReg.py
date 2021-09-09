import numpy as np
import matplotlib.pyplot as plt
import dataset as db
import LOGRegression
from MEASUREPrediction import MEASUREPrediction, showBayesPlot
from utils import split_K_folds
from stats import gauss_data

K = 5   # number of folds cross-validation

def compute_LLR_LTE(D, L, l, p, gauss=False):
    D_SETS, L_SETS = split_K_folds(D, L, K, shuffle=True)
    D = np.concatenate(D_SETS, axis=1)
    L = np.concatenate(L_SETS, axis=0)

    if gauss == True:
        for i in range(K):
            D_SETS[i] = gauss_data(D_SETS[i])

    S = []
    for i in range(K):
        DT = np.concatenate(D_SETS[:i] + D_SETS[i+1:], axis=1)
        LT = np.concatenate(L_SETS[:i] + L_SETS[i+1:], axis=0)        
        DE = D_SETS[i]

        LogReg = LOGRegression.LOGREGClass(DT, LT, l, p)
        x0 = np.zeros(D.shape[0] + 1)
        w, b, _ = LogReg.computeResult(x0)
    
        S_i = np.dot(w.T, DE) + b
        S = np.hstack((S, S_i))

    return S, L


def computeDCFMin(S,L,p):
    MP = MEASUREPrediction(p, 1.0, 1.0, S)
    MP.computeDCF(L, db.NUM_CLASSES)
    _, DCFMin = MP.getDCFMin()
    return DCFMin



def main_find_best_lambda(ptrain):
    D, L = db.load_db()
    lambdas = np.logspace(-6,1,10)
    N = lambdas.size                         
    minDCF1 = np.zeros(N)
    minDCF5 = np.zeros(N)
    minDCF9 = np.zeros(N)

    i=0
    for l in lambdas:
        LLR, LTE = compute_LLR_LTE(D, L, l, ptrain)
        minDCF1[i] = computeDCFMin(LLR, LTE, 0.1)
        LLR, LTE = compute_LLR_LTE(D, L, l, ptrain)
        minDCF5[i] = computeDCFMin(LLR, LTE, 0.5)
        LLR, LTE = compute_LLR_LTE(D, L, l, ptrain)
        minDCF9[i] = computeDCFMin(LLR, LTE, 0.9)
        i=i+1
    
    plt.figure()
    plt.semilogx(lambdas, minDCF1, label='mindcf (π=0.1)')
    plt.semilogx(lambdas, minDCF5, label='mindcf (π=0.5)')
    plt.semilogx(lambdas, minDCF9, label='mindcf (π=0.9)')
    plt.legend()
    plt.xlabel("Lambda λ")
    plt.ylabel("Min DCF")
    plt.show()


def main_find_best_threshold():
    D, L = db.load_db()

    # given optimal lambda, compute minDCF for different πt (0.1, 0.5, 0.9)
    l=1e-4

    plt.figure()
    LLR, LTE = compute_LLR_LTE(D, L, l, 0.1)
    showBayesPlot(LLR,LTE,db.NUM_CLASSES,"0.1")
    LLR, LTE = compute_LLR_LTE(D, L, l, 0.5)
    showBayesPlot(LLR,LTE,db.NUM_CLASSES,"0.5")
    LLR, LTE = compute_LLR_LTE(D, L, l, 0.9)
    showBayesPlot(LLR,LTE,db.NUM_CLASSES,"0.9")
    plt.legend()
    plt.xlabel("thres")
    plt.ylabel("DCF")
    plt.show()




if __name__ == "__main__":
    ptrain = 0.9
    main_find_best_lambda(ptrain)
    #main_find_best_threshold()

'''
OLD VALUES:

l = 1e-5

Gaussianized Features

-----  minDCF with πT (π for training) = 0.1 ---- 
mindcf (π = 0.1)  0.9832285115303985
mindcf (π = 0.5)  0.6722648883832159
mindcf (π = 0.9)  1.6865459372333367
-----  minDCF with πT (π for training) = 0.5 ---- 
mindcf (π = 0.1)  0.8378664812532263
mindcf (π = 0.5)  0.3728263961316014
mindcf (π = 0.9)  0.7959419740210486
-----  minDCF with πT (π for training) = 0.9 ---- 
mindcf (π = 0.1)  2.1520348071594873
mindcf (π = 0.5)  0.5877121456338296
mindcf (π = 0.9)  0.9015075376884422

Raw Features

-----  minDCF with πT (π for training) = 0.1 ---- 
mindcf (π = 0.1)  0.9916142557651991
mindcf (π = 0.5)  0.7562087165386682
mindcf (π = 0.9)  1.8266616099364754
-----  minDCF with πT (π for training) = 0.5 ---- 
mindcf (π = 0.1)  0.8809961758477924
mindcf (π = 0.5)  0.3727400103241575
mindcf (π = 0.9)  0.7353180999336305
-----  minDCF with πT (π for training) = 0.9 ---- 
mindcf (π = 0.1)  2.1277730370932226
mindcf (π = 0.5)  0.5745604331932197
mindcf (π = 0.9)  0.911102683227458

'''