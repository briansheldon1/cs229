import numpy as np
from svm import SVM
from utils import load_data, plot_data, plot_decision_boundary

if __name__ =="__main__":
    
    # load x and y data, plot
    x_data = load_data('data/x.dat')
    y_data = load_data('data/y.dat')
    y_data = np.where(y_data==0, -1, 1) # convert {0, 1} to {-1, 1}
    plot_data(x_data, y_data, save_to="plots/orig_data.png")

    # create and train data 
    svm = SVM(kernel="gaussian")
    svm.train(x_data, y_data, C=0.1)
    y_preds = svm.predict(x_data)
    
    # plot decision boundary
    x0_range = (np.min(x_data[:,0]), np.max(x_data[:,0]))
    x1_range = (np.min(x_data[:,1]), np.max(x_data[:,1]))
    plot_decision_boundary(svm, x0_range, x1_range, save_to="plots/svm_boundary.png")