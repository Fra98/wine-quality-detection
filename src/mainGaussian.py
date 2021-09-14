import matplotlib.pyplot as plt 
import numpy as np
import dataset as db
import GAUSSClass
from MEASUREPrediction import MEASUREPrediction, showBayesPlot
from utils import split_K_folds, computeErrorRate
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
    MP.computeDCF(L, db.NUM_CLASSES)
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

    # 1 MVG Raw
    print("MVG Raw:")
    LLR, LTE = compute_LLR_Gaussian(D, L, gauss=False, model='MVG')
    minDCF, PI, MP, actDCF = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "MVG Raw")
    MP[0].showStatsByThres(PI,LTE,2)
    print("minDCF:", minDCF, " | actDCF:", actDCF)

    # 2 TCG Raw
    print("TCG Raw:")
    LLR, LTE = compute_LLR_Gaussian(D, L, gauss=False, model='TCG')
    minDCF, PI, MP, actDCF = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "TCG Raw")
    MP[0].showStatsByThres(PI,LTE,2)
    print("minDCF:", minDCF, " | actDCF:", actDCF)

    #3 MVG Gauss
    print("MVG Gaussianized:")
    LLR, LTE = compute_LLR_Gaussian(D, L, gauss=True, model='MVG')
    minDCF, PI, MP, actDCF = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "MVG Gauss")
    MP[0].showStatsByThres(PI,LTE,2)
    print("minDCF:", minDCF, " | actDCF:", actDCF)

    plt.title('MinDCF and ActDCF comparison between different models')
    plt.savefig('./src/plots/Gaussian/Gaussian_bayes_DCF_trainSet.png')
    plt.show()


def trainModel(DTR, LTR, model='MVG'):
    GM = GAUSSClass.GAUSSClass(DTR, LTR)
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

    return GM



def computeDCF_EVAL(DTR, LTR, DTE, LTE, gauss=False, model='MVG', PCAm=None, title=''):
    # PRE-PROCESSING
    if gauss == True:
        DTR = gauss_data(DTR)
        DTE = gauss_data(DTE)

    if PCAm != None:
        DTR = PCA(DTR, PCAm)
        DTE = PCA(DTE, PCAm)

    # TRAINING
    GM = trainModel(DTR, LTR, model)

    # EVALUATION
    S = GM.computeLLR(DTE)
    
    minDCF, PI, MP, actDCF = showBayesPlot(S, LTE, db.NUM_CLASSES, title)
    MP[0].showStatsByThres(PI, LTE, 2)
    print("minDCF:", minDCF, " | actDCF:", actDCF, "| PI:", PI)


def main_evaluation():
    DTR, LTR = db.load_db(train=True)
    DTE, LTE = db.load_db(train=False)
    pi = 0.33

    plt.figure()

    print('MVG Raw')
    computeDCF_EVAL(DTR, LTR, DTE, LTE, gauss=False, model='MVG', PCAm=None, title='MVG Raw')
    
    print('TCG Raw')
    computeDCF_EVAL(DTR, LTR, DTE, LTE, gauss=False, model='TCG', PCAm=None, title='TCG Raw')

    print('MVG Gaussianized')
    computeDCF_EVAL(DTR, LTR, DTE, LTE, gauss=True, model='MVG', PCAm=None, title='MVG Gaussianized')
    
    plt.savefig('./src/plots/Gaussian/Gaussian_bayes_DCF_evalSet.png')
    

def main_comparison_EVAL_VAL():
    DTR, LTR = db.load_db(train=True)
    DTE, LTE = db.load_db(train=False)

    plt.figure()
    
    print('TCG Raw EVAL')
    computeDCF_EVAL(DTR, LTR, DTE, LTE, gauss=False, model='TCG', PCAm=None, title='TCG Raw EVAL')
    
    print("TCG Raw VAL:")
    LLR, LTE = compute_LLR_Gaussian(DTR, LTR, gauss=False, model='TCG')
    minDCF, PI, MP, actDCF = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "TCG Raw VAL")
    MP[0].showStatsByThres(PI,LTE,2)
    print("minDCF:", minDCF, " | actDCF:", actDCF, "| PI:", PI)

    plt.savefig('./src/plots/Gaussian/Gaussian_bayes_TCG_RAW_VAL_vs_EVAL.png')


if __name__ == "__main__":
    # main_DCFMin()
    # main_BayesPlot()
    # main_evaluation()
    main_comparison_EVAL_VAL()

