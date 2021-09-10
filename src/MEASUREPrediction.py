import numpy
from numpy.core.defchararray import array
from numpy.core.fromnumeric import argmax
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1)) #giro per colonna il campione

def mrow(v):
    return v.reshape(1,v.size)

class MEASUREPrediction:    
    def __init__(self,PI1, CFN, CFP, LLR):
        self.LLR=LLR
        self.CFN=CFN
        self.CFP=CFP
        self.PI1=PI1
        self.optThres=-numpy.log((PI1*CFN)/((1-PI1)*CFP))

    def computeOptDecision(self):
        return 1*(self.LLR>=self.optThres)
    
    def computeDecision(self,thres):
        return 1*(self.LLR>=thres)

    def bayes_risk(self, confm):    
        fnr=numpy.float64(confm[0,1])/numpy.float64((confm[0,1]+confm[1,1]))
        fpr=numpy.float64(confm[1,0])/numpy.float64((confm[1,0]+confm[0,0]))    
        return self.PI1*self.CFN*fnr+(1-self.PI1)*self.CFP*fpr

    def bayes_risk_norm(self, confm):    
        fnr=numpy.float64(confm[0,1])/numpy.float64((confm[0,1]+confm[1,1]))
        fpr=numpy.float64(confm[1,0])/numpy.float64((confm[1,0]+confm[0,0]))    
        dbummy=numpy.min([ (self.PI1*self.CFN),((1-self.PI1)*self.CFP)])
        drisk = self.PI1*self.CFN*fnr+(1-self.PI1)*self.CFP*fpr
        return numpy.float64(drisk)/numpy.float64(dbummy)  

    def conf_matrix(self,LTEP, LTE, nc):
        CONFM = numpy.zeros([nc,nc],dtype=numpy.int32)
        for i in range(LTEP.size):
            CONFM[LTEP[i]][LTE[i]]=CONFM[LTEP[i]][LTE[i]]+1
        return CONFM

    def showStats(self,LTEP,LTE):
        accuracy =  (LTE == LTEP).mean()*100.0
        errorRate = (100.0 - accuracy)
        print()
        print(" -- STATISTICS -- ")
        print("ACCURACY",accuracy)
        print("ERROR RATE",errorRate)

    def computeDCF(self, LTE, nc):        
        S_LLR = numpy.sort(self.LLR)  
        self.THRESH = numpy.zeros(S_LLR.shape)
        self.DCF = numpy.zeros(S_LLR.shape)
        self.TPR = numpy.zeros(S_LLR.shape)
        self.FPR = numpy.zeros(S_LLR.shape)
        for i in range(S_LLR.size):
            M_LTEP = self.computeDecision(S_LLR[i])    
            CONFM = self.conf_matrix(M_LTEP,LTE,nc)        
            DCF_CALC =  self.bayes_risk_norm(CONFM)        
            self.THRESH[i]=S_LLR[i]
            self.DCF[i]=DCF_CALC
            self.TPR[i]=1-numpy.float64(CONFM[0,1])/numpy.float64((CONFM[0,1]+CONFM[1,1]))
            self.FPR[i]=numpy.float64(CONFM[1,0])/numpy.float64((CONFM[1,0]+CONFM[0,0]))            
        return self.THRESH, self.DCF

    def computeDCF_FAST(self, LTE, nc): 
        S_MIN = min(self.LLR)
        S_MAX = max(self.LLR)

        iter = 150 
        self.DCF = numpy.zeros(iter)
        i = 0
        for tresh in numpy.linspace(S_MIN, S_MAX, iter):
            M_LTEP = self.computeDecision(tresh)    
            CONFM = self.conf_matrix(M_LTEP,LTE,nc)        
            DCF_CALC = self.bayes_risk_norm(CONFM)        
            self.DCF[i] = DCF_CALC 
            i += 1          

    def showStatsByThres(self, thresh, LTE, nc):
        M_LTEP = self.computeDecision(thresh)
        self.showStats(M_LTEP,LTE)

    def getDCFNorm(self,LTE,nc):
        LTEP = self.computeOptDecision()
        CONFM = self.conf_matrix(LTEP,LTE,nc)             
        return self.bayes_risk_norm(CONFM)    

       
    def getDCFMin(self):
        DCFMin = self.DCF.min()
        TMIN = self.DCF[self.DCF==DCFMin]
        return TMIN, DCFMin

    def showROC(self):
        plt.figure()
        plt.plot(self.FPR, self.TPR)        
        plt.show()
    

def showBayesPlot(LLR,LTE,nc,title,fast=False):
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    DCF_NORM = numpy.zeros(effPriorLogOdds.shape)
    DCF_MIN = numpy.zeros(effPriorLogOdds.shape)
    PI = numpy.zeros(effPriorLogOdds.shape)
    MPA = numpy.zeros(effPriorLogOdds.shape, dtype=object)
    for i in range(effPriorLogOdds.size):
        pi_tilde=1.0/(1.0+numpy.exp(-effPriorLogOdds[i]))        
        MP = MEASUREPrediction(pi_tilde,1,1,LLR)        
        DCF_NORM[i] = MP.getDCFNorm(LTE,nc)
        if fast:
            MP.computeDCF_FAST(LTE,nc)
        else:            
            MP.computeDCF(LTE,nc)
        [_, DCF_MIN[i]] = MP.getDCFMin()
        PI[i]=pi_tilde
        MPA[i]=MP
    plt.plot(effPriorLogOdds, DCF_NORM, label=('DCF ',title))
    plt.plot(effPriorLogOdds, DCF_MIN, label=('min DCF',title))
    plt.legend()
    plt.xlabel("Prior-log-odds")
    plt.ylabel("DCF Value")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    return min(DCF_MIN), PI[DCF_MIN==min(DCF_MIN)], MPA[DCF_MIN==min(DCF_MIN)]




    
    
    
    
    




    

