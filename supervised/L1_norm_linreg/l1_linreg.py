import numpy as np

class L1LinearRegression:
    def __init__(self):
        '''
        Initialize the L1LinearRegression model with a placeholder for the coefficients (theta).
        '''
        self.theta = None

    def train(self, X, y, lam=0.01, thresh=1e-5):
        ''' 
        Train the L1 linear regression model using coordinate descent.

        Args:
            X (np.array): Input data matrix of shape (m, n) where m is the number of examples 
                          and n is the number of features.
            y (np.array): Output vector of shape (m, 1), where m is the number of examples.
            lam (float): Regularization strength, controlling the penalty on large coefficients. Default is 0.01.
            thresh (float): Convergence threshold. The training will stop when the change in theta is less than this value. Default is 1e-5.

        Returns:
            None: Updates the model's theta based on the training data.
        '''
        # Number of features in the dataset
        n = X.shape[1]

        # Initialize the coefficients (theta) to zeros
        theta = np.zeros((n,))

        converged = False
        while not converged:
            # Store the previous value of theta to check for convergence
            prev_theta = theta.copy()

            # Update each coefficient (theta) using coordinate descent
            for i in range(n):
                # Temporarily set the coefficient to zero
                theta[i] = 0
                Xi = X[:, i]

                # Calculate the residual and the necessary quantities for the update
                a = y - X @ theta
                b = np.dot(Xi, Xi)

                # Compute the positive and negative updates for theta_i
                theta_p1 = max((Xi.T @ a - lam) / b, 0)
                theta_n1 = min((Xi.T @ a + lam) / b, 0)

                # Calculate the objective function for both possible updates
                obj_p1 = self.objective(X, Xi, theta, theta_p1, y, lam)
                obj_n1 = self.objective(X, Xi, theta, theta_n1, y, lam)

                # Select the update that minimizes the objective
                theta[i] = theta_p1 if obj_p1 < obj_n1 else theta_n1

            # Check if the model has converged by comparing the change in theta
            converged = np.sum(np.abs(theta - prev_theta)) <= thresh

        # Store the final coefficients (theta) after training
        self.theta = theta

    def predict(self, X):
        ''' 
        Make predictions using the trained model.

        Args:
            X (np.array): Input data of shape (m, n), where m is the number of examples and n is the number of features.

        Returns:
            np.array: Predicted output vector of shape (m, 1).
        '''
        return X @ self.theta

    def objective(self, X, Xi, theta, theta_i, y, lam):
        ''' 
        Compute the objective function for a given update to theta_i.

        Args:
            X (np.array): Input data matrix of shape (m, n).
            Xi (np.array): Column vector corresponding to feature i of shape (m,).
            theta (np.array): Current coefficients (m, 1).
            theta_i (float): Candidate value for the coefficient of feature i.
            y (np.array): Output vector of shape (m, 1).
            lam (float): Regularization parameter.

        Returns:
            float: Value of the objective function.
        '''
        # Calculate the residual (difference between predicted and actual values)
        a = X @ theta + Xi * theta_i - y

        # Calculate the objective function value, which includes the L1 regularization
        obj = 0.5 * np.dot(a, a) + lam * np.sum(np.abs(theta)) + lam * np.abs(theta_i)

        return obj
