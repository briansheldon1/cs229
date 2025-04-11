import numpy as np

def load_data(root):
    ''' Load data from .dat file '''
    with open(root) as f:

        # load data, split by spaces
        data = [line.strip().split(" ") for line in f]

        # remove empty strings while converting to float
        data = [np.array([float(cell) for cell in row if cell]) for row in data]

    return np.squeeze(np.array(data))


def compare_theta(theta_true, theta_pred):
    return sum(abs(theta_pred - theta_true))/len(theta_true)