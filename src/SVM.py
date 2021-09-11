import numpy
import scipy.special
import matplotlib.pyplot as plt
import scipy.optimize as scs

def mcol(v):
    return v.reshape((v.size, 1)) #giro per colonna il campione

def mrow(v):
    return v.reshape(1,v.size)    

class SVMClass:
    def __init__(self, DTR, LTR, C, K=1, pt=-1):
        self.K=K
        k_values= numpy.ones([1,DTR.shape[1]])*K
        self.D = numpy.vstack((DTR, k_values))
        self.DTR = DTR
        self.LTR = LTR
        self.Z  = mcol(2*LTR-1)
        self.C = C
        if pt!=-1:
            ptEMP = (1.0*(self.Z > 0)).sum()/(LTR.size)
            CT = C * pt / ptEMP
            CF = C * (1-pt) / (1-ptEMP)
            self.bounds = [(0,CT)] * LTR.size
            for i in range(LTR.shape[0]):
                if LTR[i]==0:
                    self.bounds[i] = (0,CF)
        else:
            self.bounds = [(0, C)] * LTR.size

        G = numpy.dot(self.D.T,self.D)
        self.H=self.Z * self.Z.T * G
                    

    def svm(self, a):         
        elle = numpy.ones(a.size) 
        f1 = 0.5*numpy.dot(numpy.dot(a.T,self.H),a)
        f2 = numpy.dot(a.T,mcol(elle))        
        v = numpy.dot(self.H,a)-elle                                        
        return f1-f2, v.T
   
    def computeResult(self, x0):
        (aopt,_,_) = scipy.optimize.fmin_l_bfgs_b(self.svm, approx_grad=False, x0=x0, iprint=0, bounds=self.bounds, factr=1.0)                
        self.W = numpy.sum(mcol(aopt)*mcol(self.Z)*self.D.T, axis=0)        
        return self.W
    
    def computeScore(self,xt):        
        k_values= numpy.ones([1,xt.shape[1]])*self.K
        xt_red = numpy.vstack((xt, k_values))                
        scores = numpy.dot(self.W.T,xt_red)
        classes = 1*(scores>0)
        return scores, classes
    
    def compute_primal_loss(self,xt,lt):
        W = mcol(self.W)  
        Z = mcol(2*lt-1)
        [scores, classes] = self.computeScore(xt)        
        fun1= 0.5 * (W*W).sum()                   
        fun2 = 1-Z*scores        
        zeros = numpy.zeros(fun2.shape)
        fun3 = self.C*numpy.sum(numpy.maximum(zeros, fun2))                                
        return fun1 +fun3


class SVMKernClass:
    def __init__(self, DTR, LTR, C, K, kf):
        self.DTR = DTR
        self.LTR = LTR
        self.C = C
        self.K = K
        self.kf = kf
        self.bounds = [(0, C)] * LTR.size
        self.Z = mcol(2 * LTR - 1)
        
        self.H = self.Z.T * self.Z * 1.0
        self.G = numpy.zeros(self.H.shape)
        for i in range(self.H.shape[0]):
            for j in range(self.H.shape[1]):
                self.G[i][j] = kf(self.DTR.T[i], self.DTR.T[j])
        self.H = self.H * self.G

    def svm(self, a):
        elle = numpy.ones(a.size)
        f1 = 0.5 * numpy.dot(numpy.dot(a.T, self.H), a)
        f2 = numpy.dot(a.T, elle.T)
        v = numpy.dot(self.H, a) - elle
        return f1 - f2, v.T

    def computeResult(self, x0):
        (aopt, fr, d) = scipy.optimize.fmin_l_bfgs_b(self.svm, approx_grad=False, x0=x0, iprint=0, bounds=self.bounds,
                                                     factr=1.0)
        self.aopt = aopt
        return aopt

    def computeScore(self, xt):
        scores = numpy.zeros(xt.shape[1])

        for i in range(xt.shape[1]):
            for j in range(self.DTR.shape[1]):
                scores[i] += self.aopt[j] * self.Z[j] * self.kf(self.DTR.T[j], xt.T[i])
        classes = 1 * (scores > 0)

        return scores.T, classes.T


def POLY_F(d, K, c):
    def kf(xi, xj):
        return (numpy.dot(xi.T, xj) + c) ** d + K

    return kf


def RBF_F(lam, K):
    def kf(xi, xj):
        norm = numpy.dot(numpy.subtract(xi, xj).T, numpy.subtract(xi, xj))
        return numpy.exp(-lam * norm) + K

    return kf






    

    
    

    
    

    
    
        
    
    



    
    
    
    
    




    

