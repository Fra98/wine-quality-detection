import matplotlib.pyplot as plt 
import numpy as np
import dataset as db
import GAUSSClass
from MEASUREPrediction import MEASUREPrediction, showBayesPlot
from utils import split_K_folds
from stats import gauss_data
from PCA import PCA

### Cross validation: 5 FOLDS
K = 5 

def compute_LLR_Gaussian(D, L, gauss=False, model='MVG', PCAm=None):
    D_SETS, L_SETS = split_K_folds(D, L, K, shuffle=True)
    D = np.concatenate(D_SETS, axis=1)
    L = np.concatenate(L_SETS, axis=0)

    # PRE-PROCESSING
    if gauss == True:
        for i in range(K):
            D_SETS[i] = gauss_data(D_SETS[i])

    if PCAm != None:
        for i in range(K):
            D_SETS[i] = PCA(D_SETS[i], PCAm)

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

        S_i = GM.computeLLR(DE)
        S = np.hstack((S, S_i))

    return S, L


def compute_DCFMin(D, L, p, gauss=False, model='MVG', PCAm=None):
    S, L = compute_LLR_Gaussian(D, L, gauss, model, PCAm)
    
    MP = MEASUREPrediction(p, 1.0, 1.0, S)
    MP.computeDCF_FAST(L, db.NUM_CLASSES)
    _, DCFMin = MP.getDCFMin()

    if gauss:
        name = f'{model} (Gaussianized) ->'
    else:
        name = f'{model} (Raw) ->'

    print(name, "DCFMin:", DCFMin)



def main_DCFMin():
    D, L = db.load_db()

    P = [0.5, 0.1, 0.9]
    for p in P:
        print("******* Pt =", p, "********")
        # Raw
        compute_DCFMin(D, L, p, gauss=False, model='MVG')
        compute_DCFMin(D, L, p, gauss=False, model='NBG')
        compute_DCFMin(D, L, p, gauss=False, model='TCG')
        compute_DCFMin(D, L, p, gauss=False, model='TCNB')

        # Gaussianized
        compute_DCFMin(D, L, p, gauss=True,  model='MVG')
        compute_DCFMin(D, L, p, gauss=True,  model='NBG')
        compute_DCFMin(D, L, p, gauss=True,  model='TCG')
        compute_DCFMin(D, L, p, gauss=True,  model='TCNB')

        # PCA Gaussianized
        compute_DCFMin(D, L, p, gauss=True,  model='MVG', PCAm=10)
        compute_DCFMin(D, L, p, gauss=True,  model='TCG', PCAm=10)

        print()


def main_BayesPlot():
    D, L = db.load_db()

    plt.figure()
    # 1
    LLR, LTE = compute_LLR_Gaussian(D, L, None, gauss=False, model='MVG')
    minDCF, PI, MP = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "MVG Raw")
    print("minDCF:", minDCF)
    print("PI =", PI)
    MP[0].showStatsByThres(PI,LTE,2)

    # 2
    LLR, LTE = compute_LLR_Gaussian(D, L, None, gauss=False, model='TCG')
    minDCF, PI, MP = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "TCG Raw")
    print("minDCF:", minDCF)
    print("PI =", PI)
    MP[0].showStatsByThres(PI,LTE,2)

    #3
    LLR, LTE = compute_LLR_Gaussian(D, L, None, gauss=True, model='MVG')
    minDCF, PI, MP = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "MVG Gauss")
    print("minDCF:", minDCF)
    print("PI =", PI)
    MP[0].showStatsByThres(PI,LTE,2)

    #4
    LLR, LTE = compute_LLR_Gaussian(D, L, None, gauss=True, model='TCG')
    minDCF, PI, MP = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "TCG Gauss")
    print("minDCF:", minDCF)
    print("PI =", PI)
    MP[0].showStatsByThres(PI,LTE,2)

    plt.title('MinDCF and ActDCF comparison between different models')
    plt.savefig('./random.png')
    plt.show()


def main_evaluation():
    D, L = db.load_db(False)

    plt.figure()
    LLR, LTE = compute_LLR_Gaussian(D, L, gauss=True, model='MVG')
    minDCF, PI, MP = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "MVG Gauss")
    print("minDCF:", minDCF)
    print("PI =", PI)
    MP[0].showStatsByThres(PI,LTE,2)
    plt.savefig('./random_EVAL')
    plt.show()

if __name__ == "__main__":
    main_DCFMin()
    # main_BayesPlot()
    # main_random()

    main_evaluation()

'''
 
 ******* Pt = 0.5 ********
MVG (Raw) -> DCFMin: 0.31239804241435565
NBG (Raw) -> DCFMin: 0.4200652528548124
TCG (Raw) -> DCFMin: 0.333605220228385
TCNB (Raw) -> DCFMin: 0.40293637846655794
MVG (Gaussianized) -> DCFMin: 0.299347471451876
NBG (Gaussianized) -> DCFMin: 0.4461663947797716
TCG (Gaussianized) -> DCFMin: 0.3474714518760196
TCNB (Gaussianized) -> DCFMin: 0.45187601957585644
MVG (Gaussianized) -> DCFMin: 0.7292006525285482
TCG (Gaussianized) -> DCFMin: 0.7887438825448614

******* Pt = 0.1 ********
MVG (Raw) -> DCFMin: 0.7781402936378468
NBG (Raw) -> DCFMin: 0.8458401305057097
TCG (Raw) -> DCFMin: 0.8123980424143556
TCNB (Raw) -> DCFMin: 0.866231647634584
MVG (Gaussianized) -> DCFMin: 0.8107667210440457
NBG (Gaussianized) -> DCFMin: 0.8205546492659055
TCG (Gaussianized) -> DCFMin: 0.7879282218597063
TCNB (Gaussianized) -> DCFMin: 0.866231647634584
MVG (Gaussianized) -> DCFMin: 1.0073409461663947
TCG (Gaussianized) -> DCFMin: 1.0057096247960848

******* Pt = 0.9 ********
MVG (Raw) -> DCFMin: 0.8425774877650898
NBG (Raw) -> DCFMin: 0.9216965742251224
TCG (Raw) -> DCFMin: 0.7487765089722676
TCNB (Raw) -> DCFMin: 0.9323001631321371
MVG (Gaussianized) -> DCFMin: 0.7895595432300164
NBG (Gaussianized) -> DCFMin: 0.8817292006525286
TCG (Gaussianized) -> DCFMin: 0.8482871125611745
TCNB (Gaussianized) -> DCFMin: 0.9306688417618271
MVG (Gaussianized) -> DCFMin: 0.99836867862969
TCG (Gaussianized) -> DCFMin: 0.99836867862969
'''
    


