import numpy as np
import matplotlib.pyplot as plt

def load_data(root):
    ''' Load data from .dat file '''
    with open(root) as f:

        # load data, split by spaces
        data = [line.strip().split(" ") for line in f]

        # remove empty strings while converting to float
        data = [[float(cell) for cell in row if cell] for row in data]

    return data

def sigmoid(z):
    '''Sigmoid function'''
    return (1/(1+np.exp(-z)))


def plot_lwlr(x_data, y_data, save_to='plots/lwlr.png', s=20):
    '''Plot lwlr data'''
    x_data0 = x_data[np.where(y_data==0)]
    x_data1 = x_data[np.where(y_data==1)]
    plt.figure()
    plt.scatter(x_data0[:,1], x_data0[:,2], color='red', label='y=0', s=s)
    plt.scatter(x_data1[:,1], x_data1[:,2], color='blue', label='y=1', s=s)
    plt.legend()
    plt.savefig(save_to)