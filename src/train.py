def MVG_FullCov():
    print("Running MVG_FullCov")


def MVG_DiagCov():
    print("Running MVG_DiagCov")


def MVG_Tied_FullCov():
    print("Running MVG_Tied_FullCov")


def MVG_Tied_DiagCov():
    print("Running MVG_Tied_DiagCov")


def LogReg():
    print("Running LogReg")


def SVM():
    print("Running SVM")


def default():
    print('Model name not valid')


MODELS = {
    "MVG_FullCov" : MVG_FullCov,
    "MVG_DiagCov"  : MVG_DiagCov,
    "MVG_Tied_FullCov"  : MVG_Tied_FullCov,
    "MVG_Tied_DiagCov"  : MVG_Tied_DiagCov,
    "LogReg" : LogReg,
    "SVM" : SVM
}


class TRAINModel:
    def __init__(self, DTR, LTR, DTE, modelName):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.modelName = modelName
         
    def run(self):
        MODELS.get(self.modelName, default)()



if __name__ == '__main__':
    TM = TRAINModel(None, None, None, "SVM")
    TM.run()
        