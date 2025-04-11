import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, sigmoid, plot_lwlr

def lwlr(X_train, y_train, x, tau, norm=0.0001):
    ''' 
        Main locally-weighted logistic regression function 
        Arguments:
            X_train (np.array): training data
            y_train (np.array): training labels
            x (np.array):       test point
            tau:                bandwidth parameter
            norm:               regularization parameter
    '''

    m, n = X_train.shape

    # Collect weights (represent how close training point is to test point)
    weights = np.exp(-np.sum((X_train - x)**2, axis=1)/(2*tau**2))

    # initialize parameters to 0
    theta = np.zeros((n,))

    # create predictions by applying sigmoid to X*Theta
    preds = sigmoid(X_train@theta)

    # Create D (diagonal matrix where D_ii = -w_i*h(x_i)*(1-h(x_i)))
    D_values = -weights*preds*(1-preds)
    D = np.diag(D_values)

    # Compute Hessian
    H = X_train.T@D@X_train - norm*np.identity(n)

    # get z where z_i = w_i(y_i - h(x_i))
    z = weights*(y_train - preds)

    # Get gradient of log likelihood
    l_grad = (X_train.T@z) - (norm*theta)

    # update 
    dTheta  = np.linalg.solve(H, l_grad)
    theta = theta - dTheta

    return theta

if __name__ == "__main__":

    # load x and y data
    x_data = load_data('data/x.dat')
    y_data = load_data('data/y.dat')

    # convert to numpy arrays
    x_data = np.array(x_data).squeeze()
    y_data = np.array(y_data).squeeze()

    # Add columnof 1's to x_data to represent x0
    ones = np.ones((x_data.shape[0],))
    x_data = np.insert(x_data, 0, ones, axis=1)
    
    # Create plot of data
    plot_lwlr(x_data, y_data, save_to='plots/orig_data.png')

    
    # Create prediction plots over entire range of (x0, x1)
    # Visualizes the decision boundary of the model
    x0_min = np.min(x_data[:,1])
    x0_max = np.max(x_data[:,1])

    x1_min = np.min(x_data[:,1])
    x1_max = np.max(x_data[:,1])

    for tau in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        x_tests = []
        y_preds = [] 
        for x0 in np.linspace(x0_min, x0_max, 50):
            for x1 in np.linspace(x1_min, x1_max, 50):

                # define test point and record
                x_test = np.array([1, x0, x1])
                x_tests.append(x_test)

                # get and save prediction
                theta = lwlr(x_data, y_data, x_test, tau)
                pred = 1 if theta@x_test > 0 else 0
                y_preds.append(pred)

        x_tests = np.array(x_tests)
        y_preds = np.array(y_preds)
        plot_lwlr(x_tests, y_preds, save_to=f'plots/lwlr_{tau}.png', s=12)

        

            