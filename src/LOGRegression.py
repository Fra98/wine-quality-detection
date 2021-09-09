import numpy
import scipy.optimize as scs
from utils import mcol, mrow
from dataset import NUM_ATTR

class LOGREGClass:
    def __init__(self, DTR, LTR, l, pt):
        self.DTR = DTR
        self.LTR = LTR
        self.l = l
        self.D0 = DTR[:, LTR == 0]
        self.D1 = DTR[:, LTR == 1]
        self.pt = pt

    def logreg_regularized(self, d):
        w, b = mrow(d[0:-1]), d[-1]
        f1 = self.l/2*(w*w).sum()
        y0 = (numpy.dot(w, self.D0) + b).reshape(-1)
        y1 = (numpy.dot(w, self.D1) + b).reshape(-1)
        f2_0 = numpy.log1p(numpy.exp(y0))
        f2_1 = numpy.log1p(numpy.exp(-y1))
        f3_1 = f2_1.mean() * self.pt
        f3_0 = f2_0.mean() * (1-self.pt)
        return f1+f3_0+f3_1

    def logreg(self, d):
        w, b = d[0:-1], d[-1]
        n = self.DTR.shape[1]
        ZLTR = 2*self.LTR - 1
        norm_squared = (w*w).sum()

        return (self.l/2)*norm_squared + (1/n)*numpy.sum(numpy.log1p(numpy.exp(-ZLTR*(numpy.dot(w.T, self.DTR) + b))))

    def computeResult(self, x0=numpy.zeros(NUM_ATTR+1)):
        [xr, fr, _] = scs.fmin_l_bfgs_b(self.logreg_regularized, x0, approx_grad=True)
        w, b = xr[0:-1], xr[-1]
        return w, b, fr
