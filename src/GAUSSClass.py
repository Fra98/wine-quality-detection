import numpy
import scipy.special as scs

from utils import mcol

class GAUSSClass:

    def __init__(self,D,L):
        self.D=D #dataset
        self.L=L #labels
    
    def computeParam(self,D):
        mu=mcol((D.mean(1)))        
        DC= D - mu 
        Eps=numpy.dot(DC,DC.T)/(float(D.shape[1]))          
        return mu, Eps
    
    def computeMVG(self):
        mu0, Eps0 = self.computeParam(self.D[:, self.L==0])
        mu1, Eps1 = self.computeParam(self.D[:, self.L==1])
        self.mu = numpy.zeros([2, mu0.shape[0], mu0.shape[1]])
        self.mu[0] = mu0
        self.mu[1] = mu1
        self.Eps = numpy.zeros([2, Eps0.shape[0], Eps0.shape[1]])
        self.Eps[0] = Eps0
        self.Eps[1] = Eps1
    
    def computeTCG(self):
        mu0, Eps0 = self.computeParam(self.D[:, self.L==0])
        mu1, Eps1 = self.computeParam(self.D[:, self.L==1])
        Eps = Eps0*self.D[:, self.L==0].shape[1]
        Eps += Eps1*self.D[:, self.L==1].shape[1]
        Eps = Eps / self.D.shape[1]
        self.mu = numpy.zeros([2, mu0.shape[0], mu0.shape[1]])
        self.mu[0]=mu0
        self.mu[1]=mu1
        self.Eps = numpy.zeros([2, Eps0.shape[0], Eps0.shape[1]])
        self.Eps[0]=Eps
        self.Eps[1]=Eps
        
    def computeNBG(self):
        self.computeMVG()
        self.Eps[0]=numpy.diag(numpy.diag(self.Eps[0]))
        self.Eps[1]=numpy.diag(numpy.diag(self.Eps[1]))

    def computeTCNB(self):
        self.computeTCG()
        self.Eps[0]=numpy.diag(numpy.diag(self.Eps[0]))
        self.Eps[1]=numpy.diag(numpy.diag(self.Eps[1]))

    def logpdf_GAU_ND(self,x,mu,Eps):
        M = x.shape[0]
        xmu_2 = x - mu
        firstTerm = -M * 0.5 * numpy.log(numpy.pi * 2)
        secondTerm = -0.5 * numpy.linalg.slogdet(Eps)[1]
        exponent = -0.5 * (xmu_2.T.dot(numpy.linalg.inv(Eps)).dot(xmu_2))
        return numpy.diag(firstTerm + secondTerm + exponent)

    def computeLLR(self,DTE,p):
        logN0=self.logpdf_GAU_ND(DTE,self.mu[0],self.Eps[0])
        logN1=self.logpdf_GAU_ND(DTE,self.mu[1],self.Eps[1])
        LC=numpy.zeros([2,DTE.shape[1]])
        LC[0]=logN0+numpy.log(p)
        LC[1]=logN1+numpy.log(1-p)
        LOGFX=scs.logsumexp(LC, axis=0)
        LOGP=numpy.zeros([2,DTE.shape[1]])
        LOGP[0]=LC[0]-LOGFX
        LOGP[1]=LC[1]-LOGFX
        return LOGP[1]-LOGP[0]
            
