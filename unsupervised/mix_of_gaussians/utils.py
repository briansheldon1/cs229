import numpy as np
import matplotlib.pyplot as plt

def generate_gauss_data(k, num_features=2, num_points=20):
    """
    Generate synthetic Gaussian-distributed data for k separate dists.
    Parameters of distributions (phi, means, covs) are randomly sampled

    Args:
        k (int): Number of Gaussian clusters to generate.
        num_features (int): Number of features (dimensions) per data point.
        num_points (int): Number of data points per cluster.

    Returns:
        np.ndarray: A list of k arrays, each of shape (num_points, num_features).
    """

    # phi (phi[j] = Prob(z_i = j))
    phi  = np.random.rand(k)
    phi /= sum(phi)

    # means of dists
    means = k*np.random.randn(k, num_features)

    # cov mats of dist
    cov_mats = []
    for j in range(k):
        A = np.random.rand(num_features, num_features)
        cov_mats.append(A.T@A + np.eye(num_features)*1e-3)

    # randomly simulate data
    data = []
    for n in range(num_points):
        j = np.random.choice(k, p=phi)
        data.append(np.random.multivariate_normal(means[j], cov_mats[j]))

    return np.array(data)


def plot_gauss_data(X: np.array, f1=0, f2=1, title="Title",outpath="out.png"):
    """ 
        Plot gaussian data

        Args:
            X (np.array): array of data mxn for m points and n features
            f1/f2 (int):  features to create 2d plot of
            outpath(str): string of where ot save plot to
    """
    plt.figure()
    plt.scatter(X[:, f1], X[:, f2])
    plt.xlabel(f"x{f1}")
    plt.ylabel(f"x{f2}")
    plt.title(title)
    plt.savefig(outpath)