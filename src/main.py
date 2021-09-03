import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import dataset as db
import GAUSSClass
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

def main(D,L, title):
    print("")
    print("----------  ", title, "  ----------" )
    print("")
    LLRMVG=0
    LLRNBG=0
    LLRTCG=0
    LLRTCNB=0
    LET=0
    for i in range(4):
        DT, LT, DE, LE = split_db_2to1(D, L, i)
        if title=="Gaussianized Features":
            gauss_func = gauss(DT)
            DT=gauss_func(DT)
            DE = gauss_func(DE)
        MVG = GAUSSClass.GAUSSClass(DT, LT)  # multi variate gaussian
        MVG.computeMVG()
        NBG = GAUSSClass.GAUSSClass(DT, LT)  # naive bayes gaussian
        NBG.computeNBG()
        TCG = GAUSSClass.GAUSSClass(DT, LT)  # tied covari gaussian
        TCG.computeTCG()
        TCNB = GAUSSClass.GAUSSClass(DT, LT)  # tied covari nay bay gaussian
        TCNB.computeTCNB()
        if i==0:
            LLRMVG = MVG.computeLLR(DE, 0.5)
            LLRNBG = NBG.computeLLR(DE, 0.5)
            LLRTCG = TCG.computeLLR(DE, 0.5)
            LLRTCNB =TCNB.computeLLR(DE, 0.5)
            LET = LE
        else:
            LLRMVG = np.hstack((LLRMVG, MVG.computeLLR(DE, 0.5)))
            LLRNBG = np.hstack((LLRNBG, NBG.computeLLR(DE, 0.5)))
            LLRTCG = np.hstack((LLRTCG, TCG.computeLLR(DE, 0.5)))
            LLRTCNB = np.hstack((LLRTCNB, TCNB.computeLLR(DE, 0.5)))
            LET = np.hstack((LET, LE))

    p = 0.9
    MPMVG = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLRMVG)
    MPNBG = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLRNBG)
    MPTCG = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLRTCG)
    MPTCNB = MEASUREPrediction.MEASUREPrediction(p, 1.0, 1.0, LLRTCNB)

    #MPMVG.showStats(MPMVG.computeOptDecision(), LET)
    print("Full-Cov ", MPMVG.getDCFNorm(LET, 2))
    #MPNBG.showStats(MPNBG.computeOptDecision(), LET)
    print("Diag-Cov ", MPNBG.getDCFNorm(LET, 2))
    #MPTCG.showStats(MPTCG.computeOptDecision(), LET)
    print("Tied Full-Cov ", MPTCG.getDCFNorm(LET, 2))
    #MPTCNB.showStats(MPTCNB.computeOptDecision(), LET)
    print("Tied Diag-Cov ",MPTCNB.getDCFNorm(LET, 2))

if __name__ == "__main__":
    D, L = db.load_db()
    main(D, L, "Raw Features")
    main(D, L, "Gaussianized Features")

'''
###########################################
pi = 0.5 
----------   Raw Features   ----------

Full-Cov  0.3637425735400666
Diag-Cov  0.44999818866830893
Tied Full-Cov  0.35134400811476596
Tied Diag-Cov  0.4341369608269333

----------   Gaussianized Features   ----------

Full-Cov  0.3106282905859054
Diag-Cov  0.4584027677148239
Tied Full-Cov  0.3653908853789306
Tied Diag-Cov  0.46414167028933007

###########################################
pi = 0.1
----------   Raw Features   ----------

Full-Cov  0.806426001062648
Diag-Cov  1.0406553398058251
Tied Full-Cov  0.9003526059025263
Tied Diag-Cov  1.0453014055933922

----------   Gaussianized Features   ----------

Full-Cov  0.8163188909819833
Diag-Cov  1.0600124378109452
Tied Full-Cov  0.8415235714630731
Tied Diag-Cov  1.1252747186398104

###########################################
pi = 0.9
----------   Raw Features   ----------

Full-Cov  1.1919739892769166
Diag-Cov  1.7905738298797278
Tied Full-Cov  0.948558179973917
Tied Diag-Cov  1.5113117664106654

----------   Gaussianized Features   ----------

Full-Cov  1.0468500941892482
Diag-Cov  1.655430372409796
Tied Full-Cov  0.9903637154035647
Tied Diag-Cov  1.6173108969714538

'''




    


