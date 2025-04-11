import numpy as np

class L1LinearRegression:
    def __init__(self):
        self.theta = None

    def train(self, X, y, lam=0.01, threshold=1e-5):
        m = X.shape[0]
        n = X.shape[1]

        # initialize theta
        theta = np.zeros((n,))

        converged = False
        while not converged:

            # store previous theta
            prev_theta = theta.copy()

            # update each theta
            for i in range(n):
                theta[i] = 0
                Xi = X[:, i]

                # si=+1
                a = y-X@theta
                b = np.dot(Xi, Xi)
                theta_p1 = max((Xi.T@a - lam)/b, 0)
                theta_n1 = min((Xi.T@a + lam)/b, 0)

                # calculate obj for theta_p1 and theta_n1
                obj_p1 = self.objective(X, Xi, theta, theta_p1, y, lam)
                obj_n1 = self.objective(X, Xi, theta, theta_n1, y, lam)

                # set theta based on best objective
                theta[i] = theta_p1 if obj_p1<obj_n1 else theta_n1

            # check if converged
            converged = sum(abs(theta-prev_theta))<=threshold

        self.theta = theta

    def predict(self, X):
        return X@self.theta

    def objective(self, X, Xi, theta, theta_i, y, lam):
        a = X@theta + Xi*theta_i - y
        obj = 0.5*np.dot(a, a) + lam*sum(abs(theta)) + lam*abs((theta_i))

        return obj