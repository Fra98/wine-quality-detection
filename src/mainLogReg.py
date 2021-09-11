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


def main_find_best_lambda(ptrain, gauss=False):
    D, L = db.load_db()
    lambdas = np.logspace(-6, 1, 10)
    N = lambdas.size
    minDCF1 = np.zeros(N)
    minDCF5 = np.zeros(N)
    minDCF9 = np.zeros(N)

    i = 0
    for l in lambdas:
        LLR, LTE = compute_LLR_LTE(D, L, l, ptrain, gauss=gauss)
        minDCF1[i] = computeDCFMin(LLR, LTE, 0.1)
        LLR, LTE = compute_LLR_LTE(D, L, l, ptrain, gauss=gauss)
        minDCF5[i] = computeDCFMin(LLR, LTE, 0.5)
        LLR, LTE = compute_LLR_LTE(D, L, l, ptrain, gauss=gauss)
        minDCF9[i] = computeDCFMin(LLR, LTE, 0.9)
        i = i + 1

    plt.figure()
    plt.semilogx(lambdas, minDCF1, label='mindcf (π=0.1)')
    plt.semilogx(lambdas, minDCF5, label='mindcf (π=0.5)')
    plt.semilogx(lambdas, minDCF9, label='mindcf (π=0.9)')
    plt.legend()
    plt.xlabel("Lambda λ")
    plt.ylabel("Min DCF")
    plt.show()


def main_find_minDCF(gauss=True):
    D, L = db.load_db()

    # given optimal lambda, compute minDCF for different πt (0.1, 0.5, 0.9)
    l = 1e-5

    LLR, LTE = compute_LLR_LTE(D, L, l, 0.1, gauss=gauss)
    print("Ptraining = 0.1")
    print("MinDCF π= 0.1", computeDCFMin(LLR, LTE, 0.1))
    print("MinDCF π= 0.5", computeDCFMin(LLR, LTE, 0.5))
    print("MinDCF π= 0.9", computeDCFMin(LLR, LTE, 0.9))
    LLR, LTE = compute_LLR_LTE(D, L, l, 0.5, gauss=gauss)
    print("Ptraining = 0.5")
    print("MinDCF π= 0.1", computeDCFMin(LLR, LTE, 0.1))
    print("MinDCF π= 0.5", computeDCFMin(LLR, LTE, 0.5))
    print("MinDCF π= 0.9", computeDCFMin(LLR, LTE, 0.9))
    LLR, LTE = compute_LLR_LTE(D, L, l, 0.9, gauss=gauss)
    print("Ptraining = 0.9")
    print("MinDCF π= 0.1", computeDCFMin(LLR, LTE, 0.1))
    print("MinDCF π= 0.5", computeDCFMin(LLR, LTE, 0.5))
    print("MinDCF π= 0.9", computeDCFMin(LLR, LTE, 0.9))


def main_find_best_threshold(gauss=True):
    D, L = db.load_db()

    # given optimal lambda, compute minDCF for different πt (0.1, 0.5, 0.9)
    l = 1e-5

    plt.figure()
    LLR, LTE = compute_LLR_LTE(D, L, l, 0.1, gauss=gauss)
    DCF1, PI1, MP1 = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "0.1", fast=True)
    MP1[0].showStatsByThres(PI1, LTE, 2)
    LLR, LTE = compute_LLR_LTE(D, L, l, 0.5, gauss=gauss)
    DCF5, PI5, MP5 = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "0.5", fast=True)
    MP5[0].showStatsByThres(PI5, LTE, 2)
    LLR, LTE = compute_LLR_LTE(D, L, l, 0.9, gauss=gauss)
    DCF9, PI9, MP9 = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "0.9", fast=True)
    MP9[0].showStatsByThres(PI9, LTE, 2)

    plt.legend()
    plt.xlabel("thres")
    plt.ylabel("DCF")
    plt.show()
    print("0.1 DCFMin: ", DCF1, " threshold: ", PI1)
    print("0.5 DCFMin: ", DCF5, " threshold: ", PI5)
    print("0.9 DCFMin: ", DCF9, " threshold: ", PI9)

if __name__ == "__main__":
    gauss=True
    ptrain = 0.9
    #main_find_best_lambda(ptrain, gauss=gauss)
    main_find_minDCF( gauss=gauss )
    #main_find_best_threshold( gauss=gauss )