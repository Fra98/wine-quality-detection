import numpy as np
import matplotlib.pyplot as plt
import dataset as db

def plot_config():
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)


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


def plot_pearsonCorrelationMatrix(PCC):
    plt.figure()
    plt.title('Pearson Correlation Coefficient Heatmap')
    plt.imshow(PCC, cmap='Greys', interpolation='nearest')
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
    plt.savefig('./src/plots/pearson.png') 
    #plt.show()

