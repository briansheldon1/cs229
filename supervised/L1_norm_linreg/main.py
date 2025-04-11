import numpy as np
from utils import load_data, compare_theta
from l1_linreg import L1LinearRegression

# load data
X = load_data("data/x.dat")  
y = load_data("data/y.dat")  
theta_true = load_data("data/theta.dat") 

# init model
model = L1LinearRegression()

# lambda values
lam_vals = [0.001, 0.01, 0.1, 0.5, 1, 5, 10]

# train on each lambda value
errors = []
for lam in lam_vals:

    # train
    model.train(X, y, lam=lam)
    
    # get error
    errors.append(compare_theta(model.theta, theta_true))

# Output the theta errors for each lambda value
for i in range(len(lam_vals)):
    print(f"Lambda = {lam_vals[i]:>6.3f} -> Theta-Error: {errors[i]:>7.5f}")
