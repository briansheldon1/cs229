import numpy as np
import matplotlib.pyplot as plt

def load_data(root):
    '''
    Load data from a .dat file.

    Parameters:
        root (str): Path to the .dat file.

    Returns:
        np.ndarray: A 2D NumPy array of data. Each row is a data point.
    '''
    with open(root) as f:
        # Read each line, strip whitespace, and split into individual strings
        data = [line.strip().split(" ") for line in f]

        # Convert each string to float, skipping empty strings
        data = np.array([np.array([float(cell) for cell in row if cell]) for row in data])

        # Remove extra singleton dimensions if any
        data = data.squeeze()

    return data


def plot_data(x_data, y_data, save_to='plots/res.png', s=20):
    '''
    Plot 2D data points with binary class labels (-1 or 1).

    Parameters:
        x_data (np.ndarray): 2D array of shape (N, 2), where each row is a 2D data point.
        y_data (np.ndarray): 1D array of shape (N,), containing labels (-1 or 1).
        save_to (str): Path to save the output plot.
        s (int): Marker size for scatter plot.
    '''
    # Extract points for each class
    x_data0 = x_data[np.where(y_data == -1)]
    x_data1 = x_data[np.where(y_data == 1)]

    # Create scatter plot
    plt.figure()
    plt.scatter(x_data0[:, 0], x_data0[:, 1], color='red', label='y = -1', s=s)
    plt.scatter(x_data1[:, 0], x_data1[:, 1], color='blue', label='y = 1', s=s)
    plt.legend()
    plt.savefig(save_to)


def plot_decision_boundary(model, x0_range, x1_range, save_to="plots/decision_boundary.png", s=12):
    '''
    Plot the decision boundary of a binary classifier over a 2D input space.

    Parameters:
        model (object): A classifier with a .predict() method that accepts a 2D NumPy array.
        x0_range (tuple): Range for x-axis (e.g., (xmin, xmax)).
        x1_range (tuple): Range for y-axis (e.g., (ymin, ymax)).
        save_to (str): Path to save the output plot.
        s (int): Marker size for decision boundary visualization.
    '''
    x_tests = []

    # Generate a grid of points in the 2D input space
    for x0 in np.linspace(*x0_range, 50):
        for x1 in np.linspace(*x1_range, 50):
            x_test = np.array([x0, x1])
            x_tests.append(x_test)

    x_tests = np.array(x_tests)

    # Predict class for each point
    y_preds = model.predict(x_tests)

    # Plot the predicted classifications as a scatter plot
    plot_data(x_tests, y_preds, save_to=save_to, s=s)