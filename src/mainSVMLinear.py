import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import dataset as db
import SVM
import MEASUREPrediction

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

def gauss(TD):
    def gauss_func(D):
        DG = np.copy(D)
        for i in range(D.shape[0]):
            for x in range(D.shape[1]):
                summ = 1*(TD[i]>D[i][x])
                DG[i][x] = (summ.sum()+1.0)/(TD.shape[1]+1.0)
        DG = norm.ppf(DG)
        return DG
    return gauss_func

def main(D,L,C,p):
    LLR=0
    LET=0
    for i in range(2):
        DT, LT, DE, LE = split_db_2to1(D, L, i)

        '''
        #uncomment this lines to get the gaussianized version
        gauss_func = gauss(DT)
        DT=gauss_func(DT)
        DE = gauss_func(DE)
        '''

        SVMOBJ = SVM.SVMClass(DT, LT, C, 10)
        x0 = np.zeros(LT.size)
        SVMOBJ.computeResult(x0)
        [score, LTEP] = SVMOBJ.computeScore(DE)
        if i==0:
            LLR = score
            LET=LE
        else:
            LLR = np.hstack((LLR, score))
            LET = np.hstack((LET, LE))

    MP = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLR)
    return MP.getDCFNorm(LET, 2)

if __name__ == "__main__":
    D, L = db.load_db()
    '''
    #compute optimal C
    ni=10
    minDCF1 = np.zeros([ni])
    minDCF5 = np.zeros([ni])
    minDCF9 = np.zeros([ni])
    i=0
    for l in np.logspace(-2,0,ni):
        minDCF1[i] = main(D, L, l, 0.1)
        minDCF5[i]=main(D, L, l, 0.5)
        minDCF9[i] = main(D, L, l, 0.9)
        i=i+1
    plt.figure()
    plt.semilogx(np.logspace(-2,0,ni), minDCF1, label='mindcf (π=0.1)')
    plt.semilogx(np.logspace(-2,0,ni), minDCF5, label='mindcf (π=0.5)')
    plt.semilogx(np.logspace(-2,0,ni), minDCF9, label='mindcf (π=0.9)')
    plt.legend()
    plt.show()
    '''
    #given optimal C, compute minDCF for different πt (0.1, 0.5, 0.9)
    C=0.1
    print("mindcf (π = 0.1) ", main(D, L, C, 0.1))
    print("mindcf (π = 0.5) ", main(D, L, C, 0.5))
    print("mindcf (π = 0.9) ", main(D, L, C, 0.9))

''' 
---- C = 0.1 ---- K-Fold with K=3 ----

Gaussianized Features
mindcf (π = 0.1)  0.957658452804084
mindcf (π = 0.5)  0.3970325135373679
mindcf (π = 0.9)  0.9302855370816536

Raw Features
mindcf (π = 0.1)  0.9854368932038835 
mindcf (π = 0.5)  0.45504639679396963
mindcf (π = 0.9)  0.9471744471744472

----- COMPARISON BETWEEN LINEAR MODELS -----

Gauss Features [Linear SVM]
mindcf (π = 0.1)  0.957658452804084
mindcf (π = 0.5)  0.3970325135373679
mindcf (π = 0.9)  0.9302855370816536

Gauss Features [Logistic Regression] 
mindcf (π = 0.1)  0.8378664812532263
mindcf (π = 0.5)  0.3728263961316014
mindcf (π = 0.9)  0.7959419740210486

Raw Features [Tied Full-Cov]
mindcf (π = 0.1)  0.9003526059025263
mindcf (π = 0.5)  0.35134400811476596
mindcf (π = 0.9)  0.948558179973917

Better one Linear: Gaussian [Tied Full-Cov]
'''