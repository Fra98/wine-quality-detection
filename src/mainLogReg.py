import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import dataset as db
import LOGRegression
import MEASUREPrediction

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 4.0 / 5.0)
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

def main(D,L,l,p):
    LLR=0
    LET=0
    for i in range(4):
        DT, LT, DE, LE = split_db_2to1(D, L, i)
        '''
        #uncomment this lines to get the gaussianized version
        gauss_func = gauss(DT)
        DT=gauss_func(DT)
        DE = gauss_func(DE)
        '''

        LogReg = LOGRegression.LOGREGClass(DT, LT, l, 0.5)
        x0 = np.zeros(D.shape[0] + 1)
        w, b, fr = LogReg.computeResult(x0)
        if i==0:
            LLR = np.dot(w.T, DE) + b
            LET=LE
        else:
            LLR = np.hstack((LLR, np.dot(w.T, DE) + b ))
            LET = np.hstack((LET, LE))

    MP = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLR)
    return MP.getDCFNorm(LET, 2)

if __name__ == "__main__":
    D, L = db.load_db()
    '''
    #compute optimal lambda = 1e-5
    minDCF1 = np.zeros([50])
    minDCF5 = np.zeros([50])
    minDCF9 = np.zeros([50])
    i=0
    for l in np.logspace(-6,1):
        minDCF1[i] = main(D, L, l, 0.1)
        minDCF5[i]=main(D, L, l, 0.5)
        minDCF9[i] = main(D, L, l, 0.9)
        i=i+1
    plt.figure()
    plt.semilogx(np.logspace(-6, 1), minDCF1, label='mindcf (π=0.1)')
    plt.semilogx(np.logspace(-6, 1), minDCF5, label='mindcf (π=0.5)')
    plt.semilogx(np.logspace(-6, 1), minDCF9, label='mindcf (π=0.9)')
    plt.legend()
    plt.show()
    '''

    #given optimal lambda, compute minDCF for different πt (0.1, 0.5, 0.9)
    l=1e-5
    print("mindcf (π = 0.1) ", main(D, L, l, 0.1))
    print("mindcf (π = 0.5) ", main(D, L, l, 0.5))
    print("mindcf (π = 0.9) ", main(D, L, l, 0.9))

'''
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