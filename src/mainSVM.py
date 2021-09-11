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

        SVMObj = SVM.SVMKernClass(DT, LT, C, 1, kf) #, 0.5)
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
        LLR, LTE = compute_LLR_LTE(D, L, C, kf, gauss=gauss)
        minDCF5[i] = computeDCFMin(LLR, LTE, 0.5)
        LLR, LTE = compute_LLR_LTE(D, L, C, kf, gauss=gauss)
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


if __name__ == "__main__":
    gauss=True
    kf = SVM.RBF_F(1.0, 1.0)
    main_find_best_C(kf,gauss)


    '''
    #given optimal C, compute minDCF for different πt (0.1, 0.5, 0.9)
    C=0.1
    print("mindcf (π = 0.1) ", main(D, L, C, 0.1))
    print("mindcf (π = 0.5) ", main(D, L, C, 0.5))
    print("mindcf (π = 0.9) ", main(D, L, C, 0.9))

    ---- C = 0.1 ---- K-Fold with K=3 ----
    
    Gaussianized Features
    mindcf (π = 0.1)  0.957658452804084
    mindcf (π = 0.5)  0.3970325135373679
    mindcf (π = 0.9)  0.9302855370816536
    
    Raw Features
    mindcf (π = 0.1)  0.9854368932038835 
    mindcf (π = 0.5)  0.45504639679396963
    mindcf (π = 0.9)  0.9471744471744472
    
    '''