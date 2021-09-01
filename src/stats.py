import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import dataset as db


def plot_histograms(D, L, prefix='original'):
    # Matrix slices selecting only attribute of a specific label
    D0 = D[:, L==0]     # LABEL = 0
    D1 = D[:, L==1]     # LABEL = 1

    for idxA in range(db.NUM_ATTR):
        plt.figure()
        plt.xlabel(db.ATTRIBUTES[idxA])
        plt.hist(D0[idxA, :], bins = 50, density = True, alpha = 0.6, label = db.LABELS[0])
        plt.hist(D1[idxA, :], bins = 50, density = True, alpha = 0.6, label = db.LABELS[1])
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig(f'./src/plots/hist_{prefix}_{idxA}.png') 
    #plt.show()


def plot_scatter(D, L, attr1, attr2):
    # Matrix slices selecting only attribute of a specific label
    D0 = D[:, L==0]     # LABEL = 0
    D1 = D[:, L==1]     # LABEL = 1
    
    if attr1 == attr2:
        exit(-1)

    plt.figure()
    plt.xlabel(db.ATTRIBUTES[attr1])
    plt.ylabel(db.ATTRIBUTES[attr2])
    plt.scatter(D0[attr1, :], D0[attr2, :], label = db.LABELS[0])
    plt.scatter(D1[attr1, :], D1[attr2, :], label = db.LABELS[1]) 
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig(f'./src/plots/scatter_{attr1}_{attr2}.png') 
    #plt.show()


def center_data(D):
    mu = D.mean(1)  # mean over all columns, for each attribute
    return D - mu.reshape(D.shape[0], 1)  # remove the mean from all points

def GAU_mu_ML(D):
    return np.mean(D, axis=1)

def GAU_var_ML(D):
    return np.var(D, axis=1)

def covarianceMatrix(D):
    N = D.shape[1] 	# number of samples
    # computing the mean for every row (attributes) and subtract it in all the dataset
    mu = D.mean(1)
    DC = D - mu.reshape(mu.size, 1)

    C = (1/N) * np.dot(DC, DC.T)

    return C

def pearsonCorrelationMatrix(D):
    return np.corrcoef(D)

def plot_pearsonCorrelationMatrix(PCC):
    plt.figure()
    plt.title('Pearson Correlation Coefficient Heatmap')
    plt.imshow(PCC, cmap='Greys', interpolation='nearest')
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig('./src/plots/pearson.png') 
    #plt.show()

def gauss_data(D):            
    DG = np.copy(D)    
    for i in range(D.shape[0]):
        for x in range(D.shape[1]):
            summ = 1*(D[i]>D[i][x])            
            DG[i][x] = (summ.sum()+1.0)/(D.shape[1]+1.0)            
    DG = norm.ppf(DG)       
    return DG


def main():
    # LOADING DATABASE
    D, L = db.load_db('./src/dataset/Train.txt')

    # PLOTTING

    # UI config
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    plot_histograms(D, L)
    # plot_scatter(D, L, 5, 3)


    # STATISTICS

    # Class distribution
    N_1 = len(L[L==1])
    N_0 = len(L[L==0])
    print(f"Number of sample of class 1 (HQ): {N_1} ({100* N_1/L.size : .2f} %)")
    print(f"Number of sample of class 0 (LQ): {N_0} ({100* N_0/L.size : .2f} %)")

    # Mean over all columns, for each attribute
    mu = D.mean(1)       

    # Data centered around mean
    DC = center_data(D)

    # Covariance matrix
    C = covarianceMatrix(D)

    # Pearson Correlation Coefficient Matrix
    PCC = pearsonCorrelationMatrix(D)
    plot_pearsonCorrelationMatrix(PCC)

    # Gaussianization
    DG = gauss_data(D)
    plot_histograms(DG, L, prefix='gaussianized')



if __name__ == "__main__":
    main()
    


