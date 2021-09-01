import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import dataset as db
import GAUSSClass
import MEASUREPrediction

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

def main(D,L,DE,LE, title):
    print(title)
    MVG = GAUSSClass.GAUSSClass(D, L)  # multi variate gaussian
    MVG.computeMVG()
    NBG = GAUSSClass.GAUSSClass(D, L)  # naive bayes gaussian
    NBG.computeNBG()
    TCG = GAUSSClass.GAUSSClass(D, L)  # tied covari gaussian
    TCG.computeTCG()
    TCNB = GAUSSClass.GAUSSClass(D, L)  # tied covari nay bay gaussian
    TCNB.computeTCNB()

    p = 0.5
    LLRMVG = MVG.computeLLR(DE, 0.5)
    LLRNBG = NBG.computeLLR(DE, 0.5)
    LLRTCG = TCG.computeLLR(DE, 0.5)
    LLRTCNB = TCNB.computeLLR(DE, 0.5)

    MPMVG = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLRMVG)
    MPNBG = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLRNBG)
    MPTCG = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLRTCG)
    MPTCNB = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLRTCNB)

    print("Full-Cov ", MPMVG.getDCFNorm(LE, 2))
    MPMVG.showStats(MPMVG.computeOptDecision(), LE)
    print("Diag-Cov ", MPNBG.getDCFNorm(LE, 2))
    MPNBG.showStats(MPNBG.computeOptDecision(), LE)
    print("Tied Full-Cov ",MPTCG.getDCFNorm(LE, 2))
    MPTCG.showStats(MPTCG.computeOptDecision(), LE)
    print("Tied Diag-Cov ",MPTCNB.getDCFNorm(LE, 2))
    MPTCNB.showStats(MPTCNB.computeOptDecision(), LE)

if __name__ == "__main__":
    D, L = db.load_db('./src/dataset/Train.txt')
    DE, LE = db.load_db('./src/dataset/Test.txt')
    gauss_func = gauss(D)
    main(D, L, DE, LE, "Raw Features")
    DG = gauss_func(D)
    DEG = gauss_func(DE)
    main(DG, L, DEG , LE, "Gaussianized Features")

'''
pi = 0.5 

Raw Features
Full-Cov  0.35942994777035603
Diag-Cov  0.3894749984393533
Tied Full-Cov  0.3368681981813263
Tied Diag-Cov  0.37154836964438065
Gaussianized Features
Full-Cov  0.35716440893106105
Diag-Cov  0.4001004016064257
Tied Full-Cov  0.3399010549971908
Tied Diag-Cov  0.39355869072143496

'''




    


