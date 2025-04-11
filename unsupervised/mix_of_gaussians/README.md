# Gaussian Mixture Model (GMM) from Scratch

This project is a simple implementation of the Expectation-Maximization (EM) 
algorithm to fit a Gaussian Mixture Model. I built it while taking 
Stanford's CS229, to better understand how the EM algorithm works 
under the hood.

## Overview

- The model is implemented in [`mix_gauss_model.py`](mix_gauss_model.py).
- To train the model, I simulate a dataset using a mixture of Gaussians.
- After training, the model generates its own simulated dataset 
  based on its learned parameters.
- I then visualize both the real and model-generated datasets 
  for comparison.

## Example Outputs

- Real training data: [`plots/real.png`](plots/real.png)
- Model-simulated data: [`plots/modelsim.png`](plots/modelsim.png)