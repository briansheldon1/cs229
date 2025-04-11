import numpy as np
import matplotlib.pyplot as plt

def load_data(root):
    ''' Load data from .dat file '''
    with open(root) as f:

        # load data, split by spaces
        data = [line.strip().split(" ") for line in f]

        # remove empty strings while converting to float
        data = np.array([np.array([float(cell) for cell in row if cell]) for row in data])

        # remove any extra dimensions
        data = data.squeeze()

    return data


def plot_data(x_data, y_data, save_to='plots/res.png', s=20):
    '''Plot data'''
    x_data0 = x_data[np.where(y_data==-1)]
    x_data1 = x_data[np.where(y_data==1)]
    plt.figure()
    plt.scatter(x_data0[:,0], x_data0[:,1], color='red', label='y=-1', s=s)
    plt.scatter(x_data1[:,0], x_data1[:,1], color='blue', label='y=1', s=s)
    plt.legend()
    plt.savefig(save_to)


def plot_decision_boundary(model, x0_range, x1_range, save_to="plots/decision_boundary.png", s=12):
    x_tests = []
    y_preds = [] 
    for x0 in np.linspace(*x0_range, 50):
        for x1 in np.linspace(*x1_range, 50):

            # define test point and record
            x_test = np.array([x0, x1])
            x_tests.append(x_test)

    x_tests = np.array(x_tests)
    y_preds = model.predict(x_tests)
    plot_data(x_tests, y_preds, save_to=save_to, s=s)