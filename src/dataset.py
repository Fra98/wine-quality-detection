import numpy as np

LABELS = {
    0 : "Low Quality",
    1 : "High Quality"
}

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

# Number of attributes
NUM_ATTR = 11


''' 
LOADING WINES DATABASE
Return:
    -D = 10D array -> Column vectors
    -L = 1D array -> labels (0 or 1)
'''
def load_db(fileName):
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
