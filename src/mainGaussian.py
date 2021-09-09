import numpy as np
import dataset as db
import GAUSSClass
from MEASUREPrediction import MEASUREPrediction
from utils import split_K_folds
from stats import gauss_data

### Cross validation: 5 FOLDS
K = 5 

def main(D, L, p, gauss=False, model='MVG'):
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
        
        GM = GAUSSClass.GAUSSClass(DT, LT)
        if model == 'MVG':
            GM.computeMVG()
        elif model =='TCG':
            GM.computeTCG()
        elif model =='NBG':
            GM.computeNBG()
        elif model=='TCNB':
            GM.computeTCNB()
        else:
            print("Incorrect model name")
            exit()

        S_i = GM.computeLLR(DE, 0.5)
        S = np.hstack((S, S_i))
    
    MP = MEASUREPrediction(p, 1.0, 1.0, S)
    # MP.computeDCF(L, db.NUM_CLASSES)
    # _, DCFMin = MP.getDCFMin()
    DCFNorm = MP.getDCFNorm(L, db.NUM_CLASSES)

    if gauss:
        str = f'{model} (Gaussianized) ->'
    else:
        str = f'{model} (Raw) ->'

    # print(str, "DCFMin:", DCFMin)
    print(str, "DCFNorm:", DCFNorm)


if __name__ == "__main__":
    D, L = db.load_db()

    main(D, L, 0.5, gauss=False, model='MVG')
    main(D, L, 0.5, gauss=False, model='NBG')
    main(D, L, 0.5, gauss=False, model='TCG')
    main(D, L, 0.5, gauss=False, model='TCNB')
    main(D, L, 0.5, gauss=True,  model='MVG')
    main(D, L, 0.5, gauss=True,  model='NBG')
    main(D, L, 0.5, gauss=True,  model='TCG')
    main(D, L, 0.5, gauss=True,  model='TCNB')


'''
OLD VALUES

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




    


