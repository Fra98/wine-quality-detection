import numpy as np
import matplotlib.pyplot as plt
import dataset as db
import SVM
from MEASUREPrediction import MEASUREPrediction, showBayesPlot
from utils import split_K_folds
from stats import gauss_data

K = 5

def computeDCFMin(S, L, p):
    MP = MEASUREPrediction(p, 1.0, 1.0, S)
    MP.computeDCF(L, db.NUM_CLASSES)
    _, DCFMin = MP.getDCFMin()
    return DCFMin

def compute_LLR_LTE(D, L, C, kf, gauss=False):
    D_SETS, L_SETS = split_K_folds(D, L, K, shuffle=True)
    D = np.concatenate(D_SETS, axis=1)
    L = np.concatenate(L_SETS, axis=0)

    if gauss == True:
        for i in range(K):
            D_SETS[i] = gauss_data(D_SETS[i])

    S = []
    for i in range(K):
        DT = np.concatenate(D_SETS[:i] + D_SETS[i + 1:], axis=1)
        LT = np.concatenate(L_SETS[:i] + L_SETS[i + 1:], axis=0)
        DE = D_SETS[i]

        SVMObj = SVM.SVMKernClass(DT, LT, C, 1, kf) #0.5
        x0 = np.zeros(LT.size)
        SVMObj.computeResult(x0)
        S_i, _ = SVMObj.computeScore(DE)
        S = np.hstack((S, S_i))

    return S, L

def main_find_best_C(kf, gauss=False):
    D, L = db.load_db()
    Csub = np.logspace(-3, 3, 10)
    N = Csub.size
    minDCF1 = np.zeros(N)
    minDCF5 = np.zeros(N)
    minDCF9 = np.zeros(N)

    i = 0
    for C in Csub:
        LLR, LTE = compute_LLR_LTE(D, L, C, kf, gauss=gauss)
        minDCF1[i] = computeDCFMin(LLR, LTE, 0.1)
        minDCF5[i] = computeDCFMin(LLR, LTE, 0.5)
        minDCF9[i] = computeDCFMin(LLR, LTE, 0.9)
        i = i + 1

    plt.figure()
    plt.semilogx(Csub, minDCF1, label='mindcf (π=0.1)')
    plt.semilogx(Csub, minDCF5, label='mindcf (π=0.5)')
    plt.semilogx(Csub, minDCF9, label='mindcf (π=0.9)')
    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Min DCF")
    plt.show()

def main_find_best_C(gauss=False):
    D, L = db.load_db()
    Csub = np.logspace(-3, 3, 10)
    N = Csub.size
    kf1 = SVM.POLY_F(2.0, 0.0, 1.0)
    minDCF1 = np.zeros(N)
    minDCF5 = np.zeros(N)
    minDCF9 = np.zeros(N)

    i = 0
    for C in Csub:
        LLR, LTE = compute_LLR_LTE(D, L, C, kf1, gauss=gauss)
        minDCF1[i] = computeDCFMin(LLR, LTE, 0.1)
        minDCF5[i] = computeDCFMin(LLR, LTE, 0.5)
        minDCF9[i] = computeDCFMin(LLR, LTE, 0.9)
        i = i + 1

    plt.figure()
    plt.semilogx(Csub, minDCF1, label='mindcf (π=0.1)')
    plt.semilogx(Csub, minDCF5, label='mindcf (π=0.5)')
    plt.semilogx(Csub, minDCF9, label='mindcf (π=0.9)')
    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Min DCF")
    plt.show()

def main_print_DCFMin(C, gauss=False):
    D, L = db.load_db()
    Csub = np.logspace(-3, 3, 10)
    kf = SVM.POLY_F(2.0, 0.0, 1.0)

    LLR, LTE = compute_LLR_LTE(D, L, C, kf, gauss=gauss)
    print("πtrain 0.1: ",computeDCFMin(LLR, LTE, 0.1))
    print("πtrain 0.5: ",computeDCFMin(LLR, LTE, 0.5))
    print("πtrain 0.9: ",computeDCFMin(LLR, LTE, 0.9))

def main_print_show_Bayes(C, kf, gauss=False):
    D, L = db.load_db()
    LLR, LTE = compute_LLR_LTE(D, L, C, kf, gauss=gauss)
    plt.figure()
    showBayesPlot(LLR,LTE,2,'Gauss Data',fast=True)
    plt.show()

if __name__ == "__main__":
    gauss=True
    main_find_best_C(gauss)
    #C = 1
    #main_print_DCFMin(C, gauss)
    #main_print_show_Bayes(C, kf, gauss)