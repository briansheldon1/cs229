import numpy as np

def load_data(root):
    ''' load data from .dat file '''
    with open(root) as f:
        # split by spaces
        data = [line.strip().split(" ") for line in f]
        # remove empty strings and convert to float
        data = [np.array([float(cell) for cell in row if cell]) for row in data]
    return np.squeeze(np.array(data))

def compare_theta(theta_true, theta_pred):
    # mean absolute difference
    return sum(abs(theta_pred - theta_true)) / len(theta_true)
