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


def main_BayesPlot(train=True):
    D, L = db.load_db(train)

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
    if train:
        plt.savefig('./src/plots/Gaussian/Gaussian_bayes_DCF_trainSet.png')
    else:
        plt.savefig('./src/plots/Gaussian/Gaussian_bayes_DCF_testSet.png')
    plt.show()

    print()


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
    computeDCF_EVAL(DTR, LTR, DTE, LTE, pi, gauss=False, model='MVG', PCAm=None, title='MVG Raw')
    
    print('TCG Raw')
    computeDCF_EVAL(DTR, LTR, DTE, LTE, pi, gauss=False, model='TCG', PCAm=None, title='TCG Raw')

    print('MVG Gaussianized')
    computeDCF_EVAL(DTR, LTR, DTE, LTE, pi, gauss=True, model='MVG', PCAm=None, title='MVG Gaussianized')
    
    plt.savefig('./src/plots/Gaussian/Gaussian_bayes_DCF_evalSet.png')
    
def main_comparison_EVAL_VAL():
    DTR, LTR = db.load_db(train=True)
    DTE, LTE = db.load_db(train=False)

    plt.figure()
    
    print('TCG Raw EVAL')
    computeDCF_EVAL(DTR, LTR, DTE, LTE, None, gauss=False, model='TCG', PCAm=None, title='TCG Raw EVAL')
    
    print("TCG Raw VAL:")
    LLR, LTE = compute_LLR_Gaussian(DTR, LTR, gauss=False, model='TCG')
    minDCF, PI, MP, actDCF = showBayesPlot(LLR, LTE, db.NUM_CLASSES, "TCG Raw VAL")
    MP[0].showStatsByThres(PI,LTE,2)
    print("minDCF:", minDCF, " | actDCF:", actDCF, "| PI:", PI)

    plt.savefig('./src/plots/Gaussian/Gaussian_bayes_TCG_RAW_VAL_vs_EVAL.png')


if __name__ == "__main__":
    # main_DCFMin()
    # main_BayesPlot(train=True)
    # main_evaluation()
    main_comparison_EVAL_VAL()

'''
DCF MIN

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



BEST MODELS TRAIN SET:

-MVG Raw    
    ACCURACY 83.19738988580751
    ERROR RATE 16.80261011419249
    minDCF: 0.31239804241435565  | actDCF: 0.3637846655791191

-TCG Raw
    ACCURACY 83.5236541598695
    ERROR RATE 16.476345840130506
    minDCF: 0.333605220228385  | actDCF: 0.3409461663947798

-MVG Gauss
    ACCURACY 85.10059815116911
    ERROR RATE 14.899401848830891
    minDCF: 0.299347471451876  | actDCF: 0.3132137030995106


BEST MODELS TEST SET:

-MVG Raw    
    ACCURACY 81.8331503841932
    ERROR RATE 18.166849615806797
    minDCF: 0.32026031587489856  | actDCF: 0.3608345298291612

-TCG Raw
    ACCURACY 85.34577387486279
    ERROR RATE 14.654226125137214
    minDCF: 0.32330617808019646  | actDCF: 0.3415657448446636

-MVG Gauss
    ACCURACY 83.91877058177826
    ERROR RATE 16.08122941822174
    minDCF: 0.32970482968142  | actDCF: 0.3336558669912812

'''
    


