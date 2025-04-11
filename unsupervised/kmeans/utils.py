import numpy as np
import matplotlib.pyplot as plt

def load_data(root):
    """
    Load 2D data from a .dat file.

    Parameters:
        root (str): Path to the .dat file.

    Returns:
        np.ndarray: Numpy array of shape (m, n), where m is the number of points.
    """
    with open(root) as f:
        # Split each line into float entries, removing empty strings
        data = [line.strip().split(" ") for line in f]
        data = [np.array([float(cell) for cell in row if cell]) for row in data]
    return np.squeeze(np.array(data))


def generate_kmeans_data(k, num_features=2, num_points=100):
    """
    Generate synthetic Gaussian-distributed data for testing K-Means.

    Parameters:
        k (int): Number of Gaussian clusters to generate.
        num_features (int): Number of features (dimensions) per data point.
        num_points (int): Number of data points per cluster.

    Returns:
        np.ndarray: A list of k arrays, each of shape (num_points, num_features).
    """
    means = 2 * k * np.random.randn(k, num_features)  # Spread-out cluster centers
    data = []

    for i in range(k):
        dist = np.random.randn(num_points, num_features) + means[i]
        data.append(dist)

    return np.array(data)


def plot_kmeans_output(X, y, k, title="title", outpath="out.png"):
    """
    Plot K-Means clustering result with color-coded clusters.

    Parameters:
        X (np.ndarray): Input data matrix of shape (m, 2).
        y (np.ndarray): Cluster assignments for each point.
        k (int): Number of clusters.
        title (str): Title of the plot.
        outpath (str): Path to save the resulting plot.
    """
    colors = ["red", "green", "blue", "pink", "yellow", "purple", "brown", "black"]

    plt.figure()
    for i in range(k):
        cluster_points = X[y == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[i % len(colors)], s=10 / k)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.savefig(outpath)
    plt.close()