import numpy
from numpy.core.defchararray import array
from numpy.core.fromnumeric import argmax, reshape
import scipy.optimize as scs
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1)) #giro per colonna il campione

def mrow(v):
    return v.reshape(1,v.size)    

class LOGREGClass:
    def __init__(self, DTR, LTR, l, pt):
        self.DTR=DTR
        self.LTR=LTR
        self.l=l
        self.D0 = DTR[:, LTR == 0]
        self.D1 = DTR[:, LTR == 1]
        self.pt = pt
    
    def logreg(self,d):
        w, b = mrow(d[0:-1]), d[-1]
        f1 = self.l/2*(w*w).sum()
        y0 = (numpy.dot(w, self.D0) + b).reshape(-1)
        y1 = (numpy.dot(w, self.D1) + b).reshape(-1)
        f2_0 = numpy.log1p(numpy.exp(y0))
        f2_1 = numpy.log1p(numpy.exp(-y1))
        f3_1 = f2_1.mean() * self.pt
        f3_0 = f2_0.mean() * (1-self.pt)
        return f1+f3_0+f3_1

    def computeResult(self,x0):
        [xr, fr ,_] = scs.fmin_l_bfgs_b(self.logreg,x0,approx_grad=True)
        w, b = xr[0:-1], xr[-1]
        return w, b, fr

    

    
    

    
    
        
    
    



    
    
    
    
    




    

