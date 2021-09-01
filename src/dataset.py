import numpy as np

# Number of classes
NUM_CLASSES = 2

LABELS = {
    0 : "Low Quality",
    1 : "High Quality"
}

# Number of attributes
NUM_ATTR = 11

ATTRIBUTES = {
    0 : "fixed acidity",
    1 : "volatile acidity",
    2 : "citric acid",
    3 : "residual sugar",
    4 : "chlorides",
    5 : "free sulfur dioxide",
    6 : "total sulfur dioxide",
    7 : "density",
    8 : "pH",
    9 : "sulphates",
    10 : "alcohol"
}


''' 
LOADING WINES DATABASE
Argument:
    -train: boolean -> TRUE : train samples, FALSE : test samples 
Return:
    -D = 10D array -> Column vectors
    -L = 1D array -> labels (0 or 1)
'''
def load_db(train=True):
    if train:
        fileName = './src/dataset/Train.txt'
    else:
        fileName = './src/dataset/Test.txt'

    list_attributes = []
    list_labels = []

    with open(fileName, "r") as inFile:    
        for line in inFile:
            fields = line.split(",")
            attributes = [float(i) for i in fields[0:NUM_ATTR]]
            label = int(fields[-1].rstrip())
            
            # Create array dim NUM_ATTRx1 for the attributes
            list_attributes.append(np.array(attributes).reshape(NUM_ATTR, 1))

            # Append to list of labels
            list_labels.append(label)

    return np.hstack(list_attributes), np.array(list_labels)
