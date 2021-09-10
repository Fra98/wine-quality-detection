import numpy as np
import scipy.special as scs
import matplotlib.pyplot as plt
import json
from utils import mcol, mrow, split_db_2to1, computeErrorRate
from stats import GAU_mu_ML, covarianceMatrix
from dataset import load_db

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    

def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]


def GAU_logpdf_MD(x, mu, C):         
    M = x.shape[0]
    s1 = -(1/2) * M * np.log(2*np.pi)
    s2 = -(1/2) * np.linalg.slogdet(C)[1]
    s3 = -(1/2) * ((np.dot((x-mu).T, np.linalg.inv(C))).T * (x-mu)).sum(axis=0)

    return s1 + s2 + s3


def GMM_logpdf(X, gmm):
    pdf = np.zeros(X.shape[1])
    for i in range(len(gmm)): 
        w_i, mu_i, C_i = gmm[i]
        s = GAU_logpdf_MD(X, mu_i, C_i) + np.log(w_i) 
        s = np.exp(s)    
        pdf += s
    
    return pdf

def GMM_joint_density(X, gmm, M):
    S = []
    for g in range(M):
        w_g, mu_g, C_g = gmm[g]
        S_g = GAU_logpdf_MD(X, mu_g, C_g)  
        S_g = S_g + np.log(w_g)    
        S.append(S_g)
    S = np.reshape(S, (M, X.shape[1]))

    return S

def marginal_densities(SJ):
    return scs.logsumexp(SJ, axis=0)


def posterior_probabilites(SJ, logdens):
    return np.exp(SJ - logdens)


def optimizeCovDegenerateCases(C, psi=0):
    U, s, _ = np.linalg.svd(C)
    s[s<psi] = psi
    C = np.dot(U, mcol(s)*U.T)
    return C


def GMM_EM(X, gmm, G, threshold, method='mvg', psi=0):
    ll_avg_old = ll_avg_new = None
    step = 0

    while(True):
        # E-STEP
        SJoint = GMM_joint_density(X, gmm, G)
        logdens = marginal_densities(SJoint)
        SPost = posterior_probabilites(SJoint, logdens)

        print("EM step:", step, "-> logdens =", np.mean(logdens))

        # Check stopping criteria
        ll_avg_new = np.mean(logdens)
        if(ll_avg_old != None and ll_avg_new-ll_avg_old < threshold):
            break

        # M-step
        Z = np.sum(SPost, axis=1)
        F = np.dot(SPost, X.T)
        mu = F / mcol(Z)
        
        C = []
        for i in range(G):
            t = SPost[i] * X
            Sg = np.dot(t, X.T)
            Cg = Sg / Z[i] - np.dot(mcol(mu[i]), mcol(mu[i]).T)   
            if method == 'naive':
                Cg = Cg * np.eye(Cg.shape[0])
            if method != 'tied':
                Cg = optimizeCovDegenerateCases(Cg, psi)  # Constraint on Cg to avoid degenerate cases
            C.append(Cg)
        wg = Z /(np.sum(Z))
        
        if method == 'tied':
            CTied = np.zeros((X.shape[0], X.shape[0]))
            for i in range(G):
                CTied += Z[i] * C[i]
            CTied = CTied / X.shape[1]
            CTied = optimizeCovDegenerateCases(CTied, psi)
            C = [CTied] * len(C)

        for i in range(G):
            gmm[i] = (wg[i], mcol(mu[i]), C[i])

        ll_avg_old = ll_avg_new
        step += 1

    return gmm, SPost, logdens


def GMM_LBG(X, G=2, alpha=0.1, threshold=10**(-6), method='mvg', psi=0):
    mu = mcol(GAU_mu_ML(X))
    C = covarianceMatrix(X)
    C = optimizeCovDegenerateCases(C, psi)  # Constraint on initial GMM Covariances to avoid degenerate cases
    gmm = [(1.0, mu, C)]

    numIter = int(np.log2(G))
    for _ in range(0, numIter):
        def split_GMM(gmm_i):
            w_i, mu_i, C_i = gmm_i
            U, s, Vh = np.linalg.svd(C_i)
            d = U[:, 0:1] * s[0]**0.5 * alpha
            gmm_i_1 = (w_i/2, mu_i+d, C_i)
            gmm_i_2 = (w_i/2, mu_i-d, C_i)
            return gmm_i_1, gmm_i_2
            
        new_gmm = []
        for gmm_i in gmm:
            gmm_i_1, gmm_i_2 = split_GMM(gmm_i)
            new_gmm.append(gmm_i_1)
            new_gmm.append(gmm_i_2)
        gmm = new_gmm

    return GMM_EM(X, gmm, G, threshold, method)


'''
    Methods:
        - 'mvg'     -> MVG
        - 'tied'    -> Tied Covariance MVG
        - 'naive'   -> Naive Bayes MVG
'''
class GMMClass():
    def __init__(self, G, threshold, alpha, psi, method):
        self.G = G
        self.threshold = threshold
        self.alpha = alpha
        self.psi = psi
        self.method = method

    def compute_GMM_LBG(self, DTR):
        return GMM_LBG(DTR, self.G, self.alpha, self.threshold, self.method, self.psi)


def GMM_classify(DTR, LTR, DTE, LTE, pi_1, G, threshold, alpha, psi, method):
    ## TRAINING

    # Classify every sample of the training set to his respective class
    DTR0 = DTR[:, LTR==0]
    DTR1 = DTR[:, LTR==1]

    GMMOBJ = GMMClass(G, threshold, alpha, psi, method)

    gmm0, _, _ = GMMOBJ.compute_GMM_LBG(DTR0)
    gmm1, _, _ = GMMOBJ.compute_GMM_LBG(DTR1)


    ## TESTING

    S_DTE0_log = GMM_logpdf(DTE, gmm0)
    S_DTE1_log = GMM_logpdf(DTE, gmm1)

    # Score matrix: S[i,j] = conditional probability for sample j given class i
    S = np.vstack([S_DTE0_log, S_DTE1_log])

    # Compute matrix of joint log-probabilites SJoint_log 
    # multiply every row of S by the prior log-probability of corresponding class
    # n.b: in our dataset is 1/3 for every class, but now is calculated generally
    P_C_prior_log = np.log([1-pi_1, pi_1]).reshape(2, 1)
    SJoint_log = S + P_C_prior_log

    # Compute marginal log-densities for all classes
    MD_log = marginal_densities(SJoint_log)                 
    
    # Compute class posterior log-probabilities
    SPost_log = posterior_probabilites(SJoint_log, MD_log)   # SPost_log[i,j] = posterior log-probability for sample j given class i

    # Compute predicted labels
    CLTE = np.argmax(SPost_log, axis=0)                     

    ## EVALUATION OF THE MODEL
    computeErrorRate(CLTE, LTE)


def main_LBG_EM():
    D, _ = load_db(True)
    G = 4
    threshold = 10**(-6)
    alpha = 0.1
    psi = 0.01
    method = 'mvg'

    GMMOBJ = GMMClass(G, threshold, alpha, psi, method)
    gmm, SPost, logdens = GMMOBJ.compute_GMM_LBG(D)
    print(gmm)

    return


def main_classification():
    DTR, LTR = load_db(train=True)
    DTE, LTE = load_db(train=False)
    
    # Hyper-paramaterss
    pi_1 = 0.5
    G = 8
    threshold = 10**(-6)
    alpha = 0.1
    psi = 0.01
    method = 'mvg'

    GMM_classify(DTR, LTR, DTE, LTE, pi_1, G, threshold, alpha, psi, method)


if __name__ == '__main__':
    # main_LBG_EM()
    main_classification()
