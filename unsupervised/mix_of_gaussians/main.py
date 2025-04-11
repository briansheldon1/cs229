from mix_gauss_model import MixGauss
from utils import generate_gauss_data, plot_gauss_data
import numpy as np

if __name__ == "__main__":
    # Number of Gaussian clusters to simulate
    k = 5
    num_points = 1000

    # Generate real data (training data)
    X = generate_gauss_data(k, num_points=num_points)

    # Create model and train on data
    model = MixGauss()
    model.train(X, k=k)

    # Generate simulated data from trained model
    X_sim = model.create_sim_data(num_points)

    # Plot real and simulated data
    plot_gauss_data(X, title="Real Data", outpath="plots/real.png")
    plot_gauss_data(X_sim, title="Model-Simulated Data", 
                                       outpath="plots/modelsim.png")