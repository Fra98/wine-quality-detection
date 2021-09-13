import matplotlib.pyplot as plt 
from dataset import load_db, NUM_CLASSES
from MEASUREPrediction import showBayesPlot
from mainGMM import compute_GMM_LLR_EVAL_rec


def main_compare():
    DTR, LTR = load_db(train=True)
    DTE, LTE = load_db(train=False)

    GMMmod = {
        "G" : 8,
        "alpha" : 0.1,
        "gauss" : True,
        "model" : 'TCG',
        "PCAm" : None,
        "title" : 'GMM TCG Gaussianized 8G (Calibrated)',
    }

    # Recalibration
    pt = 0.5

    plt.figure()

    LLR, _, LTE = compute_GMM_LLR_EVAL_rec(DTR, LTR, DTE, LTE, GMMmod["G"], GMMmod["alpha"], pt, GMMmod["gauss"], GMMmod["model"], GMMmod["PCAm"])
    minDCF, PI, MP, actDCF = showBayesPlot(LLR, LTE, NUM_CLASSES, GMMmod['title'], False, 'red')
    MP[0].showStatsByThres(PI, LTE, 2)
    print("minDCF:", minDCF, " | actDCF:", actDCF, "| PI:", PI)

    plt.savefig('./src/plots/final.png')
    plt.show()



if __name__ == '__main__':
    main_compare()
