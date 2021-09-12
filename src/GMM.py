import numpy as np
import scipy.special as scs
import json
from utils import mcol
from stats import GAU_mu_ML, covarianceMatrix


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


def GMM_EM(X, gmm, G, threshold, method='MVG', psi=0):
    ll_avg_old = ll_avg_new = None
    step = 0

    while(True):
        # E-STEP
        SJoint = GMM_joint_density(X, gmm, G)
        logdens = marginal_densities(SJoint)
        SPost = posterior_probabilites(SJoint, logdens)

        # print("EM step:", step, "-> logdens =", np.mean(logdens))

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
            C.append(Cg)
        wg = Z /(np.sum(Z))
        
        if method=='MVG' or method=='NBG':
            for i in range(G):
                if method == 'NBG':
                    C[i] = C[i] * np.eye(C[i].shape[0])
                C[i] = optimizeCovDegenerateCases(C[i], psi)

        if method=='TCG' or method=='TCNB':
            CTied = np.zeros((X.shape[0], X.shape[0]))
            for i in range(G):
                CTied += Z[i] * C[i]
            CTied = CTied / X.shape[1]
            if method == 'TCNB':
                 CTied = CTied * np.eye(CTied.shape[0])
            CTied = optimizeCovDegenerateCases(CTied, psi)
            C = [CTied] * len(C)

        for i in range(G):
            gmm[i] = (wg[i], mcol(mu[i]), C[i])

        ll_avg_old = ll_avg_new
        step += 1

    return gmm, SPost, logdens


def GMM_LBG(X, G, threshold=10**(-6), alpha=0.1, psi=0, method='MVG'):
    mu = mcol(GAU_mu_ML(X))
    C = covarianceMatrix(X)
    C = optimizeCovDegenerateCases(C, psi)  # Constraint on initial GMM Covariances to avoid degenerate cases
    if method == 'NBG' or method == 'TCNB':
        C = C * np.eye(C.shape[0])
    gmm = [(1.0, mu, C)]

    if G == 1:
        SJoint = GMM_joint_density(X, gmm, G)
        logdens = marginal_densities(SJoint)
        SPost = posterior_probabilites(SJoint, logdens)
        return gmm, SPost, logdens

    numIter = int(np.log2(G))
    tmpG = 1
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
        tmpG = tmpG * 2
        gmm, SPost, logdens = GMM_EM(X, new_gmm, tmpG, threshold, method, psi)

    return gmm, SPost, logdens


'''
    Methods:
        - 'MVG'    -> MVG
        - 'TCG'    -> Tied Covariance MVG
        - 'NBG'    -> Naive Bayes MVG
        - 'TCNB'   -> Tied Covariance Naive Bayes MVG
'''
class GMMClass():
    def __init__(self, DTR, LTR, G, threshold, alpha, psi, method):
        self.DTR = DTR
        self.LTR = LTR
        self.G = G
        self.threshold = threshold
        self.alpha = alpha
        self.psi = psi
        self.method = method
    
    def computeGMMs(self):
        DT0 = self.DTR[:, self.LTR==0]
        DT1 = self.DTR[:, self.LTR==1]
        self.gmm0, _, _ = GMM_LBG(DT0, self.G, self.threshold, self.alpha, self.psi, self.method)
        self.gmm1, _, _ = GMM_LBG(DT1, self.G, self.threshold, self.alpha, self.psi, self.method)

    def computeLLR(self, DTE):
        S0 = GMM_logpdf(DTE, self.gmm0)
        S1 = GMM_logpdf(DTE, self.gmm1)

        return S1-S0