import random as rand
import numpy as np

def train_test_split(X: np.array, y: np.array, split=0.7):

    # define num training examples
    m = len(X)
    m_train = round(m*split)-1

    # shuffle up total indices
    indices = [i for i in range(m)]
    rand.shuffle(indices)

    # slice shuffled idnices to train and test
    train_indices = indices[:m_train]
    test_indices = indices[m_train:]
    
    # slice X and y on training and testing indices
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


