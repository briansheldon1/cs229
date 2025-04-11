import numpy as np
import matplotlib.pyplot as plt

from k_means import KMeans
from utils import load_data, generate_kmeans_data, plot_kmeans_output


# === CONFIGURATION ===
CS229_DATA_PATH = "data/X.dat"
CS229_PLOT_DIR = "plots/cs229_data"
SIM_PLOT_DIR = "plots/sim_data"


def run_cs229_experiment():
    """
    Loads real data from CS229 problem set and runs KMeans clustering
    for different values of k. Plots the results.
    """
    X = load_data(CS229_DATA_PATH)
    k_values = [3, 4]
    model = KMeans()

    for k in k_values:
        model.train(X, k=k, n_init=5)
        y_pred = model.predict(X)

        plot_kmeans_output(
            X, y_pred, k,
            title=f"K-Means (k={k}) on CS229 Data",
            outpath=f"{CS229_PLOT_DIR}/k={k}.png"
        )


def run_simulation_experiment(k: int):
    """
    Generates synthetic data from k Gaussian clusters, fits KMeans,
    and plots the results.
    """
    data = generate_kmeans_data(k, num_points=200)
    X = np.vstack(data)

    model = KMeans()
    model.train(X, k=k, n_init=3)
    y_pred = model.predict(X)

    plot_kmeans_output(
        X, y_pred, k,
        title=f"K-Means (k={k}) on Simulated Data",
        outpath=f"{SIM_PLOT_DIR}/k={k}.png"
    )


if __name__ == "__main__":
    run_cs229_experiment()

    for k in range(2, 9):
        run_simulation_experiment(k)
