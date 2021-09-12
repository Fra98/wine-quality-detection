import matplotlib.pyplot as plt 
import numpy as np
from GMM import GMMClass
from dataset import load_db, NUM_CLASSES
from MEASUREPrediction import MEASUREPrediction, showBayesPlot
from utils import split_K_folds
from stats import gauss_data
from PCA import PCA

# Fixed paramaters 
K = 5       # Number of K-folds
threshold = 10**(-6)
psi = 0.01


def compute_GMM_LLR(D, L, G, alpha, gauss=False, model='MVG', PCAm=None):
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
        
        GMMOBJ = GMMClass(DT, LT, G, threshold, alpha, psi, model)
        GMMOBJ.computeGMMs()
        
        S_i = GMMOBJ.computeLLR(DE)
        S = np.hstack((S, S_i))

    return S, L


def compute_GMM_DCFMin(D, L, p, G, alpha, gauss=False, model='MVG', PCAm=None):
    S, L = compute_GMM_LLR(D, L, G, alpha, gauss, model, PCAm)
    
    MP = MEASUREPrediction(p, 1.0, 1.0, S)
    MP.computeDCF(L, NUM_CLASSES)
    _, DCFMin = MP.getDCFMin()

    if gauss:
        strgauss = 'Gau'
    else:
        strgauss = 'Raw'

    print(f'MinDCF: {DCFMin:.4f} -> {model} ({strgauss}) (p={p}) (alpha={alpha}) G={G}')

    return DCFMin


def main_tuning_alpha():
    D, L = load_db()
    p = 0.5
    G = 16
    
    alphas =  np.logspace(-3, 0, 20)
    N = alphas.size
    minDCF_MVG = np.zeros(N)
    minDCF_NBG = np.zeros(N)
    minDCF_TCG = np.zeros(N)
    minDCF_TCNB = np.zeros(N)

    i = 0
    for alpha in alphas:
        minDCF_MVG[i] = compute_GMM_DCFMin(D, L, p, G, alpha, True, 'MVG')
        minDCF_NBG[i] = compute_GMM_DCFMin(D, L, p, G, alpha, True, 'NBG')
        minDCF_TCG[i] = compute_GMM_DCFMin(D, L, p, G, alpha, True, 'TCG')
        minDCF_TCNB[i] = compute_GMM_DCFMin(D, L, p, G, alpha, True, 'TCNB')
        i += 1

    plt.figure()
    plt.semilogx(alphas, minDCF_MVG, label='mindcf MVG')
    plt.semilogx(alphas, minDCF_NBG, label='mindcf NBG')
    plt.semilogx(alphas, minDCF_TCG, label='mindcf TCG')
    plt.semilogx(alphas, minDCF_TCNB, label='mindcf TCNB')
    plt.legend()
    plt.xlabel("Alpha (α)")
    plt.ylabel("Min DCF")
    plt.savefig('./src/plots/GMM/gmm_tuning_alpha.png')


def plot_MinDCF_GMM(title, minDCF_RAW, minDCF_GAU, G):
    X = np.arange(len(G))
    plt.figure()
    ax = plt.subplot(111)
    plt.bar(X-0.2, minDCF_RAW, 0.4, color='orange', label='Raw')    
    plt.bar(X+0.2, minDCF_GAU, 0.4, color='red', label='Gauss')
    plt.xticks(X, G)
    plt.title(title)
    plt.xlabel("GMM components")
    plt.ylabel('Min DCF')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f'./src/plots/GMM/gmm_componentsMinDCF_{title}.png')

def main_find_best_G():
    D, L = load_db()
    p = 0.5
    alpha = 0.1
    G = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    MODELS = ['MVG', 'NBG', 'TCG', 'TCNB']
    

    for model in MODELS:
        MINDCF_RAW = np.zeros(len(G))
        MINDCF_GAU = np.zeros(len(G))
        i = 0
        for g in G:
            MINDCF_RAW[i] = compute_GMM_DCFMin(D, L, p, g, alpha, False, model)
            MINDCF_GAU[i] = compute_GMM_DCFMin(D, L, p, g, alpha, True, model)
            i += 1
        print(f'({model}) Raw -> minDCF = {MINDCF_RAW.min()}, G={G[np.where(MINDCF_RAW == MINDCF_RAW.min())[0][0]]}')
        print(f'({model}) Gau -> minDCF = {MINDCF_GAU.min()}, G={G[np.where(MINDCF_GAU == MINDCF_GAU.min())[0][0]]}')
        plot_MinDCF_GMM(model, MINDCF_RAW, MINDCF_GAU, G)

def main_best_models():
    D, L = load_db()
    alpha = 0.1
    
    for p in [0.5, 0.1]:
        compute_GMM_DCFMin(D, L, p, 512, alpha, False, 'MVG')
        compute_GMM_DCFMin(D, L, p, 128, alpha, True, 'MVG')
        compute_GMM_DCFMin(D, L, p, 128, alpha, True, 'TCG')


def main_BayesPlot(train=True):
    D, L = load_db(train)
    alpha = 0.1

    plt.figure()

    # 1 MVG Raw
    print("MVG Raw 512G (alpha=0.1) (pt=0.5):")
    LLR, LTE = compute_GMM_LLR(D, L, 512, alpha, False, 'MVG')
    minDCF, PI, MP, actDCF = showBayesPlot(LLR, LTE, NUM_CLASSES, "MVG Raw 512G (alpha=0.1) (pt=0.5)", True)
    MP[0].showStatsByThres(PI,LTE,2)
    print("minDCF:", minDCF, " | actDCF:", actDCF)

    # 2 TCG Gaussianized
    print("MVG Gau 128G (alpha=0.1) (pt=0.5):")
    LLR, LTE = compute_GMM_LLR(D, L, 128, alpha, True, 'TCG')
    minDCF, PI, MP, actDCF = showBayesPlot(LLR, LTE, NUM_CLASSES, "MVG Gaussianized 128G (alpha=0.1) (pt=0.5)", False)
    MP[0].showStatsByThres(PI,LTE,2)
    print("minDCF:", minDCF, " | actDCF:", actDCF)

    if train:
        plt.savefig('./src/plots/GMM/GMM_bayes_DCF_trainSet.png')
    else:
        plt.savefig('./src/plots/GMM/GMM_bayes_DCF_testSet.png')
    plt.show()

    print()

if __name__ == '__main__':
    # main_tuning_alpha()
    # main_find_best_G()
    # main_best_models()
    main_BayesPlot()

'''
# BEST MODELS

(MVG) Raw -> minDCF = 0.283, G=512, pt=0.5
(MVG) Gau -> minDCF = 0.300, G=128, pt-0.5
(TCG) Gau -> minDCF = 0.285, G=128, pt=0.5

The Naive Bayes assumption doesn't perform well both with or without tied covariance.
Gaussanization it's harmful in this case.
Non-diagonal models, on the other hand, provide smaller minimum DCF at higher values of G. 
The tied covariance model perform better with Gaussianization while the non-tied with raw
data. 

'''