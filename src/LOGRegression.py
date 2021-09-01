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
    def __init__(self, DTR, LTR, l):      
        self.DTR=DTR
        self.LTR=LTR
        self.l=l
    
    def logreg(self,d):
        w, b = mrow(d[0:-1]), d[-1] 
        z = 2*self.LTR-1        
        f1 = self.l/2*(w*w).sum()
        y = (numpy.dot(w,self.DTR)+b).reshape(-1)
        f2 = numpy.log1p(numpy.exp(-z*y))
        f3 = f2.mean()
        return f1+f3    

    def computeResult(self,x0):
        [xr, fr ,_] = scs.fmin_l_bfgs_b(self.logreg,x0,approx_grad=True,
                                    factr=5000, maxfun=20000)
        w, b = xr[0:-1], xr[-1]
        return w, b, fr

    

    
    

    
    
        
    
    



    
    
    
    
    




    

