import numpy as np
import matplotlib.pyplot as plt
import dataset as db
import SVM
from MEASUREPrediction import MEASUREPrediction, showBayesPlot, recalScores
from utils import split_K_folds, mcol
from stats import gauss_data

K = 5

def computeDCFMin(S, L, p):
    MP = MEASUREPrediction(p, 1.0, 1.0, S)
    MP.computeDCF(L, db.NUM_CLASSES)
    _, DCFMin = MP.getDCFMin()
    return DCFMin

def computeDCFActual(S, L, p):
    MP = MEASUREPrediction(p, 1.0, 1.0, S)
    LTE = MP.computeOptDecision()
    MP.showStats(LTE,L)
    return MP.getDCFNorm(L, db.NUM_CLASSES)

def compute_LLR_LTE_eval(C, kf, ps, pt=-1.0,  gauss=False):
    DTR, LTR = db.load_db(True)
    DTE, LTE = db.load_db(False)

    if gauss == True:
        DTR = gauss_data(DTR)
        DTE = gauss_data(DTE)

    SVMObj = SVM.SVMKernClass(DTR, LTR, C, 1, kf, pt)
    x0 = np.zeros(LTR.size)
    SVMObj.computeResult(x0)
    STE, _ = SVMObj.computeScore(DTE)
    STR, _ = SVMObj.computeScore(DTR)
    STEC = recalScores(STR, LTR, STE, ps)

    return STEC, STE, LTE

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

        SVMObj = SVM.SVMKernClass(DT, LT, C, 1, kf)
        x0 = np.zeros(LT.size)
        SVMObj.computeResult(x0)
        S_i, _ = SVMObj.computeScore(DE)
        S = np.hstack((S, S_i))

    return S, L

def compute_LLR_LTE_rec(D, L, C, kf, ps, pt, gauss=False):
    D_SETS, L_SETS = split_K_folds(D, L, K, shuffle=True)
    D = np.concatenate(D_SETS, axis=1)
    L = np.concatenate(L_SETS, axis=0)

    if gauss == True:
        for i in range(K):
            D_SETS[i] = gauss_data(D_SETS[i])

    S = []
    ST = []
    for i in range(K):
        DT = np.concatenate(D_SETS[:i] + D_SETS[i + 1:], axis=1)
        LT = np.concatenate(L_SETS[:i] + L_SETS[i + 1:], axis=0)
        DE = D_SETS[i]

        SVMObj = SVM.SVMKernClass(DT, LT, C, 1, kf, pt)
        x0 = np.zeros(LT.size)
        SVMObj.computeResult(x0)
        S_i, _ = SVMObj.computeScore(DE)
        S_Train, _ = SVMObj.computeScore(DT)
        ST_i = recalScores(S_Train, LT, S_i, ps)
        S = np.hstack((S, S_i))
        ST = np.hstack((ST, ST_i))

    return ST, S, L

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

def main_find_best_C_and_L(gauss=False):
    D, L = db.load_db()
    Csub = np.logspace(-3, 3, 10)
    N = Csub.size
    kf1 = SVM.RBF_F(np.exp(-1), 1.0)
    kf2 = SVM.RBF_F(np.exp(-2), 1.0)
    kf3 = SVM.RBF_F(np.exp(-3), 1.0)
    minDCF1 = np.zeros(N)
    minDCF2 = np.zeros(N)
    minDCF3 = np.zeros(N)

    i = 0
    for C in Csub:
        LLR, LTE = compute_LLR_LTE(D, L, C, kf1, gauss)
        minDCF1[i] = computeDCFMin(LLR, LTE, 0.5)
        LLR, LTE = compute_LLR_LTE(D, L, C, kf2, gauss)
        minDCF2[i] = computeDCFMin(LLR, LTE, 0.5)
        LLR, LTE = compute_LLR_LTE(D, L, C, kf3, gauss)
        minDCF3[i] = computeDCFMin(LLR, LTE, 0.5)
        i = i + 1

    plt.figure()
    plt.semilogx(Csub, minDCF1, label='log(λ)=-1')
    plt.semilogx(Csub, minDCF2, label='log(λ)=-2')
    plt.semilogx(Csub, minDCF3, label='log(λ)=-3')
    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Min DCF")
    plt.show()

def main_print_DCFMin(C, kf, gauss=False):
    D, L = db.load_db()
    Csub = np.logspace(-3, 3, 10)
    N = Csub.size

    LLR, LTE = compute_LLR_LTE(D, L, C, kf, gauss=gauss)
    print("πtilde 0.1: ",computeDCFMin(LLR, LTE, 0.1))
    print("πtilde 0.5: ",computeDCFMin(LLR, LTE, 0.5))
    print("πtilde 0.9: ",computeDCFMin(LLR, LTE, 0.9))

    print("πtilde 0.1: ", computeDCFActual(LLR, LTE, 0.1))
    print("πtilde 0.5: ", computeDCFActual(LLR, LTE, 0.5))
    print("πtilde 0.9: ", computeDCFActual(LLR, LTE, 0.9))

def main_print_DCFMin_eval(C, kf, ps, pt, gauss=False):
    D, L = db.load_db()
    Csub = np.logspace(-3, 3, 10)
    N = Csub.size

    LLRC, LLR, LTE = compute_LLR_LTE_eval(C, kf, ps, pt, gauss=gauss)
    print("πtilde 0.1: ",computeDCFMin(LLR, LTE, 0.1))
    print("πtilde 0.5: ",computeDCFMin(LLR, LTE, 0.5))
    print("πtilde 0.9: ",computeDCFMin(LLR, LTE, 0.9))


    print("πtilde 0.1 (cal): ", computeDCFActual(LLRC, LTE, 0.1))
    print("πtilde 0.5 (cal): ", computeDCFActual(LLRC, LTE, 0.5))
    print("πtilde 0.9 (cal): ", computeDCFActual(LLRC, LTE, 0.9))


def main_print_DCF_recal(C, kf, ps, pt, gauss=False):
    D, L = db.load_db()
    Csub = np.logspace(-3, 3, 10)
    N = Csub.size

    LLRC_TR, LLR_TR, LTE_TR = compute_LLR_LTE_rec(D, L, C, kf, ps, pt, gauss=gauss)
    LLRC_TE, LLR_TE, LTE_TE = compute_LLR_LTE_eval(C, kf, ps, pt, gauss=gauss)

    plt.figure()
    showBayesPlot(LLRC_TR, LTE_TR, 2, 'SVM RBF [VAL] (Calibrated)', color='blue')
    showBayesPlot(LLRC_TE, LTE_TE, 2, 'SVM RBF [EVAL] (Calibrated)', color='red')
    plt.show()

    #print("Non Calibrated")
    #print("πtilde 0.1: ", computeDCFActual(LLR, LTE, 0.1))
    #print("πtilde 0.5: ", computeDCFActual(LLR, LTE, 0.5))
    #print("πtilde 0.9: ", computeDCFActual(LLR, LTE, 0.9))

def compute_SVM_LLR_EVAL_rec(C, lamb, ps, gauss):
    pt=-1
    kf=SVM.RBF_F(np.exp(lamb),1)
    LLRC, LLR, LTE = compute_LLR_LTE_eval(C, kf, ps, pt, gauss=gauss)
    return LLRC, LLR, LTE

if __name__ == "__main__":
    '''
    #find best C and l for the gauss / non - gauss datasets
    gauss=False
    main_find_best_C_and_L(gauss)   
        
    #print different DCF min for validation dataset    
    gauss = True
    kf = SVM.RBF_F(np.exp(-1), 1.0)
    C=2
    #gauss = False
    #kf = SVM.RBF_F(np.exp(-3), 1.0)
    #C=10
    #main_print_DCFMin(C, kf, gauss)
        
    #recalibration time!
    #pt = 0.5  #di train
    #main_print_DCF_recal(C,kf, pt=pt, gauss=gauss)
    '''

    #print different DCF min for evaluation dataset
    gauss = True
    kf = SVM.RBF_F(np.exp(-1), 1.0)
    C=2
    '''
    gauss = False
    kf = SVM.RBF_F(np.exp(-3), 1.0)
    C=10
    '''
    pt = -1
    ps = 0.5
    main_print_DCFMin_eval(C, kf, ps, pt, gauss)
    #main_print_DCF_recal(C, kf, ps, pt, gauss=gauss)

